#include "runtime/ps2_pad.h"
#include "runtime/ps2_pad_host.h"
#include "ps2_host_backend.h"
#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <strings.h>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{
    constexpr uint8_t kPadAnalogMarker = 0x73;
    constexpr uint8_t kPadStickCenter = 0x80;
    constexpr uint32_t kPadStickNeutral = 0x80808080u;
    constexpr int kGamepad = 0;

    // Drop a latch the guest never came back for, so a tap during a long
    // non-polling stretch (loading, cutscene) does not fire much later.
    constexpr uint64_t kLatchTimeoutMs = 1000u;

    constexpr uint16_t PAD_LEFT = 0x0080u;
    constexpr uint16_t PAD_DOWN = 0x0040u;
    constexpr uint16_t PAD_RIGHT = 0x0020u;
    constexpr uint16_t PAD_UP = 0x0010u;
    constexpr uint16_t PAD_START = 0x0008u;
    constexpr uint16_t PAD_R3 = 0x0004u;
    constexpr uint16_t PAD_L3 = 0x0002u;
    constexpr uint16_t PAD_SELECT = 0x0001u;
    constexpr uint16_t PAD_SQUARE = 0x8000u;
    constexpr uint16_t PAD_CROSS = 0x4000u;
    constexpr uint16_t PAD_CIRCLE = 0x2000u;
    constexpr uint16_t PAD_TRIANGLE = 0x1000u;
    constexpr uint16_t PAD_R1 = 0x0800u;
    constexpr uint16_t PAD_L1 = 0x0400u;
    constexpr uint16_t PAD_R2 = 0x0200u;
    constexpr uint16_t PAD_L2 = 0x0100u;

    // Published by the render thread in ps2PadPollHost(), consumed on the EE thread.
    std::atomic<bool> g_hostPolled{false};
    std::atomic<uint32_t> g_held{0u};              // active-high PAD_* mask
    std::atomic<uint32_t> g_latched{0u};           // press edges not consumed yet
    std::atomic<uint32_t> g_sticks{kPadStickNeutral}; // packed rx,ry,lx,ly
    std::atomic<uint64_t> g_latchStampMs{0u};

    struct KeyBinding
    {
        int key;
        uint16_t mask;
    };

    constexpr KeyBinding kKeyBindings[] = {
        {KEY_UP, PAD_UP},
        {KEY_DOWN, PAD_DOWN},
        {KEY_LEFT, PAD_LEFT},
        {KEY_RIGHT, PAD_RIGHT},
        {KEY_X, PAD_CROSS}, {KEY_SPACE, PAD_CROSS},
        {KEY_C, PAD_CIRCLE}, {KEY_ESCAPE, PAD_CIRCLE},
        {KEY_Z, PAD_SQUARE}, {KEY_KP_0, PAD_SQUARE},
        {KEY_V, PAD_TRIANGLE}, {KEY_KP_1, PAD_TRIANGLE},
        {KEY_Q, PAD_L1},
        {KEY_E, PAD_R1},
        {KEY_LEFT_SHIFT, PAD_L2},
        {KEY_RIGHT_SHIFT, PAD_R2},
        {KEY_ENTER, PAD_START},
        {KEY_TAB, PAD_SELECT},
        {KEY_R, PAD_L3},
        {KEY_F, PAD_R3},
    };

    struct PadBinding
    {
        int button;
        uint16_t mask;
    };

    constexpr PadBinding kPadBindings[] = {
        {GAMEPAD_BUTTON_LEFT_FACE_UP, PAD_UP},
        {GAMEPAD_BUTTON_LEFT_FACE_DOWN, PAD_DOWN},
        {GAMEPAD_BUTTON_LEFT_FACE_LEFT, PAD_LEFT},
        {GAMEPAD_BUTTON_LEFT_FACE_RIGHT, PAD_RIGHT},
        {GAMEPAD_BUTTON_RIGHT_FACE_DOWN, PAD_CROSS},
        {GAMEPAD_BUTTON_RIGHT_FACE_RIGHT, PAD_CIRCLE},
        {GAMEPAD_BUTTON_RIGHT_FACE_LEFT, PAD_SQUARE},
        {GAMEPAD_BUTTON_RIGHT_FACE_UP, PAD_TRIANGLE},
        {GAMEPAD_BUTTON_LEFT_TRIGGER_1, PAD_L1},
        {GAMEPAD_BUTTON_RIGHT_TRIGGER_1, PAD_R1},
        {GAMEPAD_BUTTON_LEFT_TRIGGER_2, PAD_L2},
        {GAMEPAD_BUTTON_RIGHT_TRIGGER_2, PAD_R2},
        {GAMEPAD_BUTTON_MIDDLE_RIGHT, PAD_START},
        {GAMEPAD_BUTTON_MIDDLE_LEFT, PAD_SELECT},
        {GAMEPAD_BUTTON_LEFT_THUMB, PAD_L3},
        {GAMEPAD_BUTTON_RIGHT_THUMB, PAD_R3},
    };

    uint8_t axisToByte(float axis)
    {
        if (axis > -0.125f && axis < 0.125f)
            return kPadStickCenter;
        const float mapped = 128.0f + axis * (axis < 0.0f ? 128.0f : 127.0f);
        return static_cast<uint8_t>(mapped < 0.0f ? 0.0f : (mapped > 255.0f ? 255.0f : mapped));
    }

    uint8_t keyboardAxis(int negative, int positive, uint8_t fallback)
    {
        const bool low = IsKeyDown(negative);
        const bool high = IsKeyDown(positive);
        return low || high ? (low == high ? kPadStickCenter : low ? 0u : 255u) : fallback;
    }

    uint64_t nowMs()
    {
        using namespace std::chrono;
        return static_cast<uint64_t>(
            duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count());
    }

    uint32_t keyMask(int key)
    {
        uint32_t mask = 0u;
        for (const KeyBinding &binding : kKeyBindings)
        {
            if (binding.key == key)
            {
                mask |= binding.mask;
            }
        }
        return mask;
    }

    // drainQueue must only be true on the render thread: GetKeyPressed() mutates
    // raylib's queue, while every other call here is a plain read.
    void sampleHost(bool drainQueue, uint32_t &held, uint32_t &pressed, uint32_t &sticks)
    {
        held = 0u;
        pressed = 0u;
        sticks = kPadStickNeutral;

        if (IsGamepadAvailable(kGamepad))
        {
            for (const PadBinding &binding : kPadBindings)
            {
                if (IsGamepadButtonDown(kGamepad, binding.button))
                {
                    held |= binding.mask;
                }
                if (IsGamepadButtonPressed(kGamepad, binding.button))
                {
                    pressed |= binding.mask;
                }
            }

            const uint8_t rx = axisToByte(GetGamepadAxisMovement(kGamepad, GAMEPAD_AXIS_RIGHT_X));
            const uint8_t ry = axisToByte(GetGamepadAxisMovement(kGamepad, GAMEPAD_AXIS_RIGHT_Y));
            const uint8_t lx = axisToByte(GetGamepadAxisMovement(kGamepad, GAMEPAD_AXIS_LEFT_X));
            const uint8_t ly = axisToByte(GetGamepadAxisMovement(kGamepad, GAMEPAD_AXIS_LEFT_Y));
            sticks = (static_cast<uint32_t>(rx) << 24) | (static_cast<uint32_t>(ry) << 16) |
                     (static_cast<uint32_t>(lx) << 8) | static_cast<uint32_t>(ly);
        }

        for (const KeyBinding &binding : kKeyBindings)
        {
            if (IsKeyDown(binding.key))
            {
                held |= binding.mask;
            }
        }

        const uint8_t rx = keyboardAxis(KEY_J, KEY_L, static_cast<uint8_t>(sticks >> 24u));
        const uint8_t ry = keyboardAxis(KEY_I, KEY_K, static_cast<uint8_t>(sticks >> 16u));
        const uint8_t lx = keyboardAxis(KEY_A, KEY_D, static_cast<uint8_t>(sticks >> 8u));
        const uint8_t ly = keyboardAxis(KEY_W, KEY_S, static_cast<uint8_t>(sticks));
        sticks = (uint32_t(rx) << 24u) | (uint32_t(ry) << 16u) | (uint32_t(lx) << 8u) | ly;

        if (drainQueue)
        {
            // raylib queues every GLFW press, including one released again inside
            // the same poll, which IsKeyPressed() would already have missed.
            for (int key = GetKeyPressed(); key != 0; key = GetKeyPressed())
            {
                pressed |= keyMask(key);
            }
        }
    }
}

namespace
{
    // Scripted input. DQ8_PAD_SCRIPT is either a file path or the script text
    // itself, entries separated by ';' or newlines:
    //
    //     <guest-frame> <BUTTON>[+<BUTTON>...] [hold-frames]
    //     <guest-frame> STICK <rx>,<ry>,<lx>,<ly> [hold-frames]
    //
    // STICK sets both analog sticks (0-255, 128 = centre) for the hold.
    // Timed in guest vsync ticks so a script replays the same way whether the
    // game is running at 3 fps or 60. Lines starting with '#' are comments.
    struct PadScriptEvent
    {
        uint64_t frame = 0u;
        uint64_t holdFrames = 4u;
        uint32_t mask = 0u;
        uint32_t sticks = kPadStickNeutral;
        bool hasSticks = false;
        bool active = false;
        bool delivered = false;
    };

    std::atomic<uint64_t> g_guestFrame{0u};
    // Appended to by DQ8_PAD_LIVE on the render thread, read by the EE thread.
    std::mutex g_padScriptMutex;
    std::vector<PadScriptEvent> g_padScript;
    bool g_padScriptVerbose = false;

    uint32_t padScriptButtonMask(const std::string &name)
    {
        static const std::unordered_map<std::string, uint32_t> kNames = {
            {"UP", PAD_UP}, {"DOWN", PAD_DOWN}, {"LEFT", PAD_LEFT}, {"RIGHT", PAD_RIGHT},
            {"CROSS", PAD_CROSS}, {"CIRCLE", PAD_CIRCLE}, {"SQUARE", PAD_SQUARE},
            {"TRIANGLE", PAD_TRIANGLE}, {"START", PAD_START}, {"SELECT", PAD_SELECT},
            {"L1", PAD_L1}, {"R1", PAD_R1}, {"L2", PAD_L2}, {"R2", PAD_R2},
            {"L3", PAD_L3}, {"R3", PAD_R3},
        };
        std::string upper;
        upper.reserve(name.size());
        for (const char c : name)
        {
            upper.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
        }
        const auto it = kNames.find(upper);
        return it == kNames.end() ? 0u : it->second;
    }

    // Caller holds g_padScriptMutex (or runs before any other thread can see the script).
    void parsePadScript(const std::string &text, uint64_t base = 0u)
    {
        std::string entry;
        std::istringstream stream(text);
        while (std::getline(stream, entry, '\n'))
        {
            std::string chunk;
            std::istringstream entries(entry);
            while (std::getline(entries, chunk, ';'))
            {
                const size_t hash = chunk.find('#');
                if (hash != std::string::npos)
                {
                    chunk.erase(hash);
                }
                std::istringstream fields(chunk);
                uint64_t frame = 0u;
                std::string buttons;
                if (!(fields >> frame >> buttons))
                {
                    continue;
                }
                PadScriptEvent event{};
                event.frame = base + frame;
                if (strcasecmp(buttons.c_str(), "STICK") == 0)
                {
                    std::string values;
                    unsigned rx = 0u, ry = 0u, lx = 0u, ly = 0u;
                    if (!(fields >> values) ||
                        std::sscanf(values.c_str(), "%u,%u,%u,%u", &rx, &ry, &lx, &ly) != 4)
                    {
                        std::fprintf(stderr, "[pad] bad STICK entry in DQ8_PAD_SCRIPT: '%s'\n",
                                     chunk.c_str());
                        continue;
                    }
                    event.hasSticks = true;
                    event.sticks = ((rx & 0xFFu) << 24u) | ((ry & 0xFFu) << 16u) |
                                   ((lx & 0xFFu) << 8u) | (ly & 0xFFu);
                    uint64_t hold = 0u;
                    if (fields >> hold && hold > 0u)
                    {
                        event.holdFrames = hold;
                    }
                    g_padScript.push_back(event);
                    continue;
                }
                uint64_t hold = 0u;
                if (fields >> hold && hold > 0u)
                {
                    event.holdFrames = hold;
                }
                size_t start = 0u;
                while (start <= buttons.size())
                {
                    const size_t plus = buttons.find('+', start);
                    const std::string name = buttons.substr(
                        start, plus == std::string::npos ? std::string::npos : plus - start);
                    const uint32_t mask = padScriptButtonMask(name);
                    if (mask == 0u && !name.empty())
                    {
                        std::fprintf(stderr, "[pad] unknown button in DQ8_PAD_SCRIPT: '%s'\n",
                                     name.c_str());
                    }
                    event.mask |= mask;
                    if (plus == std::string::npos)
                    {
                        break;
                    }
                    start = plus + 1u;
                }
                if (event.mask != 0u)
                {
                    g_padScript.push_back(event);
                }
            }
        }
        std::stable_sort(g_padScript.begin(), g_padScript.end(),
                  [](const PadScriptEvent &l, const PadScriptEvent &r) { return l.frame < r.frame; });
    }

    void ensurePadScriptLoaded()
    {
        static const bool loaded = [] {
            const char *value = std::getenv("DQ8_PAD_SCRIPT");
            g_padScriptVerbose = std::getenv("DQ8_PAD_SCRIPT_VERBOSE") != nullptr;
            if (value == nullptr || *value == '\0')
            {
                return true;
            }
            std::lock_guard<std::mutex> lock(g_padScriptMutex);
            std::ifstream file(value);
            if (file)
            {
                std::ostringstream contents;
                contents << file.rdbuf();
                parsePadScript(contents.str());
            }
            else
            {
                parsePadScript(value);
            }
            std::fprintf(stderr, "[pad] DQ8_PAD_SCRIPT: %zu events\n", g_padScript.size());
            return true;
        }();
        (void)loaded;
    }

    // DQ8_PAD_LIVE=path is a script fed while the game runs: lines appended to
    // the file use the DQ8_PAD_SCRIPT format, with the frame counted from the
    // guest frame at which the line is read, so a tool can drive the game a
    // step at a time from screenshots.
    void pollPadLive(uint64_t frame)
    {
        static const std::string path = [] {
            const char *value = std::getenv("DQ8_PAD_LIVE");
            return std::string(value != nullptr ? value : "");
        }();
        static std::streamoff consumed = 0;
        static uint64_t lastPoll = 0u;
        if (path.empty() || (frame < lastPoll + 4u && lastPoll != 0u))
        {
            return;
        }
        lastPoll = frame;
        std::ifstream file(path, std::ios::binary);
        if (!file)
        {
            return;
        }
        file.seekg(0, std::ios::end);
        const std::streamoff size = file.tellg();
        if (size < consumed)
        {
            consumed = 0;
        }
        if (size == consumed)
        {
            return;
        }
        file.seekg(consumed);
        std::string text(static_cast<size_t>(size - consumed), '\0');
        file.read(text.data(), static_cast<std::streamsize>(text.size()));
        // Only whole lines; a writer may be midway through the last one.
        const size_t end = text.rfind('\n');
        if (end == std::string::npos)
        {
            return;
        }
        text.resize(end + 1u);
        consumed += static_cast<std::streamoff>(text.size());
        std::lock_guard<std::mutex> lock(g_padScriptMutex);
        parsePadScript(text, frame);
        if (g_padScriptVerbose)
        {
            std::fprintf(stderr, "[pad] frame %llu: live script read\n",
                         static_cast<unsigned long long>(frame));
        }
    }

    // Merges scripted buttons into what the guest is about to read, so a human
    // can still take over while a script is running. Applied at the guest's
    // pad read rather than on the render thread, so a replay sees the press at
    // the same guest frame it was recorded at. A press whose whole window fell
    // between two guest reads is still handed over once, at the next read.
    void applyPadScript(uint64_t frame, uint32_t &active, uint32_t &sticks)
    {
        ensurePadScriptLoaded();
        std::lock_guard<std::mutex> lock(g_padScriptMutex);
        for (PadScriptEvent &event : g_padScript)
        {
            if (frame < event.frame)
            {
                break;
            }
            const bool on = frame < event.frame + event.holdFrames;
            if (on || !event.delivered)
            {
                active |= event.mask;
                if (event.hasSticks)
                {
                    sticks = event.sticks;
                }
                if (!event.active && g_padScriptVerbose)
                {
                    std::fprintf(stderr, "[pad] frame %llu: press 0x%04x\n",
                                 static_cast<unsigned long long>(frame), event.mask);
                }
                event.delivered = true;
            }
            event.active = on;
        }
    }

    // DQ8_PAD_RECORD=path writes the buttons the guest reads, in the
    // DQ8_PAD_SCRIPT format, so a session played by hand can be replayed with
    // DQ8_PAD_SCRIPT=path. One line per button press or stick position,
    // written when it ends.
    class PadRecorder
    {
    public:
        PadRecorder()
        {
            const char *value = std::getenv("DQ8_PAD_RECORD");
            if (value == nullptr || *value == '\0')
            {
                return;
            }
            m_file = std::fopen(value, "w");
            if (m_file == nullptr)
            {
                std::fprintf(stderr, "[pad] DQ8_PAD_RECORD: cannot open '%s'\n", value);
                return;
            }
            std::fprintf(m_file, "# DQ8_PAD_RECORD: <guest-frame> <button> <hold-frames>\n");
            std::fflush(m_file);
            std::fprintf(stderr, "[pad] DQ8_PAD_RECORD: recording to %s\n", value);
        }

        ~PadRecorder()
        {
            if (m_file != nullptr)
            {
                // Buttons still down at exit, so the tail of a session is kept.
                record(m_lastFrame + 1u, 0u, kPadStickNeutral);
                std::fclose(m_file);
            }
        }

        void record(uint64_t frame, uint32_t active, uint32_t sticks)
        {
            if (m_file == nullptr)
            {
                return;
            }
            m_lastFrame = frame;
            if (sticks != m_sticks)
            {
                if (m_sticks != kPadStickNeutral)
                {
                    std::fprintf(m_file, "%llu STICK %u,%u,%u,%u %llu\n",
                                 static_cast<unsigned long long>(m_sticksAt), m_sticks >> 24u,
                                 (m_sticks >> 16u) & 0xFFu, (m_sticks >> 8u) & 0xFFu, m_sticks & 0xFFu,
                                 static_cast<unsigned long long>(std::max<uint64_t>(1u, frame - m_sticksAt)));
                }
                m_sticks = sticks;
                m_sticksAt = frame;
            }
            const uint32_t changed = active ^ m_active;
            if (changed == 0u)
            {
                std::fflush(m_file);
                return;
            }
            for (uint32_t bit = 1u; bit <= 0x8000u; bit <<= 1u)
            {
                if ((changed & bit) == 0u)
                {
                    continue;
                }
                const int index = __builtin_ctz(bit);
                if ((active & bit) != 0u)
                {
                    m_downAt[index] = frame;
                }
                else
                {
                    std::fprintf(m_file, "%llu %s %llu\n",
                                 static_cast<unsigned long long>(m_downAt[index]), buttonName(bit),
                                 static_cast<unsigned long long>(std::max<uint64_t>(1u, frame - m_downAt[index])));
                }
            }
            std::fflush(m_file);
            m_active = active;
        }

    private:
        static const char *buttonName(uint32_t bit)
        {
            switch (bit)
            {
            case PAD_UP: return "UP";
            case PAD_DOWN: return "DOWN";
            case PAD_LEFT: return "LEFT";
            case PAD_RIGHT: return "RIGHT";
            case PAD_CROSS: return "CROSS";
            case PAD_CIRCLE: return "CIRCLE";
            case PAD_SQUARE: return "SQUARE";
            case PAD_TRIANGLE: return "TRIANGLE";
            case PAD_START: return "START";
            case PAD_SELECT: return "SELECT";
            case PAD_L1: return "L1";
            case PAD_R1: return "R1";
            case PAD_L2: return "L2";
            case PAD_R2: return "R2";
            case PAD_L3: return "L3";
            default: return "R3";
            }
        }

        std::FILE *m_file = nullptr;
        uint32_t m_active = 0u;
        uint64_t m_downAt[16]{};
        uint64_t m_lastFrame = 0u;
        uint32_t m_sticks = kPadStickNeutral;
        uint64_t m_sticksAt = 0u;
    };

    // Shared tail of both host paths: publish held state and latch edges long
    // enough that a tap between two guest polls is not lost.
    void publishHostState(uint32_t held, uint32_t pressed, uint32_t sticks)
    {
        pollPadLive(g_guestFrame.load(std::memory_order_relaxed));
        g_held.store(held);
        g_sticks.store(sticks);

        const uint64_t now = nowMs();
        if (pressed != 0u)
        {
            if (g_latched.fetch_or(pressed) == 0u)
            {
                g_latchStampMs.store(now);
            }
        }
        else
        {
            const uint32_t stale = g_latched.load();
            if (stale != 0u && (now - g_latchStampMs.load()) > kLatchTimeoutMs)
            {
                g_latched.fetch_and(~stale);
            }
        }

        g_hostPolled.store(true);
    }
}

void ps2PadPublishHostState(uint32_t held, uint32_t pressed, uint32_t sticks)
{
    publishHostState(held, pressed, sticks);
}

void ps2PadSetGuestFrame(uint64_t frame)
{
    // Set by both the present loop and the guest's pad read; never step back.
    uint64_t current = g_guestFrame.load(std::memory_order_relaxed);
    while (current < frame &&
           !g_guestFrame.compare_exchange_weak(current, frame, std::memory_order_relaxed))
    {
    }
}

uint64_t ps2PadCurrentGuestFrame()
{
    return g_guestFrame.load(std::memory_order_relaxed);
}

void ps2PadPollHost()
{
    if (!IsWindowReady())
    {
        return;
    }

    uint32_t held = 0u;
    uint32_t pressed = 0u;
    uint32_t sticks = kPadStickNeutral;
    sampleHost(true, held, pressed, sticks);
    publishHostState(held, pressed, sticks);
}

bool PSPadBackend::readState(int /*port*/, int /*slot*/, uint8_t *data, size_t size)
{
    if (!data || size < 32)
        return false;

    std::memset(data, 0, 32);
    data[0] = 0x01;
    data[1] = kPadAnalogMarker;
    data[2] = 0xFF;
    data[3] = 0xFF;
    data[4] = data[5] = data[6] = data[7] = kPadStickCenter;

    uint32_t active = 0u;
    uint32_t sticks = kPadStickNeutral;
    if (g_hostPolled.load())
    {
        // exchange() so each latched press is handed to the guest exactly once.
        active = g_held.load() | g_latched.exchange(0u);
        sticks = g_sticks.load();
    }
    else
    {
        // No host present loop (embedders that never call ps2PadPollHost).
        uint32_t pressed = 0u;
        sampleHost(false, active, pressed, sticks);
    }

    const uint64_t frame = g_guestFrame.load(std::memory_order_relaxed);
    applyPadScript(frame, active, sticks);
    static PadRecorder recorder;
    recorder.record(frame, active, sticks);

    data[4] = static_cast<uint8_t>(sticks >> 24);
    data[5] = static_cast<uint8_t>(sticks >> 16);
    data[6] = static_cast<uint8_t>(sticks >> 8);
    data[7] = static_cast<uint8_t>(sticks);

    const uint16_t btns = static_cast<uint16_t>(0xFFFFu & ~active);
    data[2] = static_cast<uint8_t>(btns & 0xFF);
    data[3] = static_cast<uint8_t>(btns >> 8);
    return true;
}
