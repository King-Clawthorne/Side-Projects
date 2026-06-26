using UnityEngine;

namespace SmallGame
{
    /// <summary>
    /// Generates all sound effects procedurally at runtime — no audio files are
    /// shipped or loaded. Clips are synthesized once on first access into float
    /// PCM buffers and cached, then played through a pooled set of AudioSources.
    ///
    /// The manager is self-bootstrapping: the first call to <see cref="Instance"/>
    /// creates a hidden, scene-persistent GameObject, so no manual scene wiring or
    /// prefab setup is required.
    /// </summary>
    public class SfxManager : MonoBehaviour
    {
        const int SampleRate = 44100;
        const int VoiceCount = 8;

        static SfxManager _instance;
        public static SfxManager Instance
        {
            get
            {
                if (_instance == null)
                {
                    var go = new GameObject("SfxManager") { hideFlags = HideFlags.HideAndDontSave };
                    _instance = go.AddComponent<SfxManager>();
                    DontDestroyOnLoad(go);
                }
                return _instance;
            }
        }

        AudioSource[] _voices;
        int _nextVoice;

        AudioClip _bounce, _switch, _powerup, _death;

        void Awake()
        {
            if (_instance != null && _instance != this) { Destroy(gameObject); return; }
            _instance = this;

            _voices = new AudioSource[VoiceCount];
            for (int i = 0; i < VoiceCount; i++)
            {
                var src = gameObject.AddComponent<AudioSource>();
                src.playOnAwake = false;
                src.spatialBlend = 0f; // 2D
                _voices[i] = src;
            }

            _bounce = BuildBounce();
            _switch = BuildSwitch();
            _powerup = BuildPowerup();
            _death = BuildDeath();
        }

        void OnDestroy()
        {
            if (_instance == this) _instance = null;
        }

        // ---- Public play hooks -------------------------------------------------

        public void PlayBounce() => Play(_bounce, 0.45f, RandomPitch(0.06f));
        public void PlaySwitch() => Play(_switch, 0.5f, RandomPitch(0.04f));
        public void PlayPowerup() => Play(_powerup, 0.5f, 1f);
        public void PlayDeath() => Play(_death, 0.6f, 1f);

        void Play(AudioClip clip, float volume, float pitch)
        {
            if (clip == null || _voices == null) return;
            var src = _voices[_nextVoice];
            _nextVoice = (_nextVoice + 1) % _voices.Length;
            src.pitch = pitch;
            src.PlayOneShot(clip, volume);
        }

        // Deterministic-ish per-call variation without relying on a seeded RNG.
        float RandomPitch(float spread) => 1f + Random.Range(-spread, spread);

        // ---- Synthesis ---------------------------------------------------------

        // Short upward "boing" — a sine whose frequency sweeps up, with a fast
        // percussive attack and exponential decay.
        AudioClip BuildBounce()
        {
            float dur = 0.18f;
            int n = (int)(SampleRate * dur);
            var data = new float[n];
            float phase = 0f;
            for (int i = 0; i < n; i++)
            {
                float t = (float)i / n;
                float freq = Mathf.Lerp(260f, 660f, EaseOut(t));
                phase += freq * 2f * Mathf.PI / SampleRate;
                float env = Attack(t, 0.01f) * Mathf.Exp(-5f * t);
                data[i] = Mathf.Sin(phase) * env;
            }
            return ToClip("sfx_bounce", data);
        }

        // Bright "zap" for a color switch — square-ish tone sweeping down with a
        // touch of detuned shimmer.
        AudioClip BuildSwitch()
        {
            float dur = 0.22f;
            int n = (int)(SampleRate * dur);
            var data = new float[n];
            float p1 = 0f, p2 = 0f;
            for (int i = 0; i < n; i++)
            {
                float t = (float)i / n;
                float freq = Mathf.Lerp(880f, 330f, t);
                p1 += freq * 2f * Mathf.PI / SampleRate;
                p2 += freq * 1.5f * 2f * Mathf.PI / SampleRate;
                float square = Mathf.Sign(Mathf.Sin(p1)) * 0.5f + Mathf.Sin(p2) * 0.3f;
                float env = Attack(t, 0.005f) * Mathf.Exp(-6f * t);
                data[i] = square * env * 0.7f;
            }
            return ToClip("sfx_switch", data);
        }

        // Cheerful rising arpeggio for power-ups / coins (major triad + octave).
        AudioClip BuildPowerup()
        {
            float[] notes = { 523.25f, 659.25f, 783.99f, 1046.5f }; // C5 E5 G5 C6
            float noteDur = 0.07f;
            int nPer = (int)(SampleRate * noteDur);
            int n = nPer * notes.Length;
            var data = new float[n];
            for (int k = 0; k < notes.Length; k++)
            {
                float phase = 0f;
                for (int i = 0; i < nPer; i++)
                {
                    float t = (float)i / nPer;
                    phase += notes[k] * 2f * Mathf.PI / SampleRate;
                    float env = Attack(t, 0.02f) * Mathf.Exp(-3.5f * t);
                    // Sine + soft third harmonic for a little sparkle.
                    float s = Mathf.Sin(phase) + 0.25f * Mathf.Sin(phase * 3f);
                    data[k * nPer + i] = s * env * 0.5f;
                }
            }
            return ToClip("sfx_powerup", data);
        }

        // Falling, noisy "thud" for death — descending tone mixed with decaying
        // noise.
        AudioClip BuildDeath()
        {
            float dur = 0.55f;
            int n = (int)(SampleRate * dur);
            var data = new float[n];
            float phase = 0f;
            for (int i = 0; i < n; i++)
            {
                float t = (float)i / n;
                float freq = Mathf.Lerp(420f, 70f, EaseOut(t));
                phase += freq * 2f * Mathf.PI / SampleRate;
                float tone = Mathf.Sin(phase);
                float noise = Random.value * 2f - 1f;
                float env = Attack(t, 0.01f) * Mathf.Exp(-4f * t);
                data[i] = (tone * 0.6f + noise * 0.4f * Mathf.Exp(-8f * t)) * env;
            }
            return ToClip("sfx_death", data);
        }

        // ---- Helpers -----------------------------------------------------------

        static float Attack(float t, float attackFrac)
            => t < attackFrac ? t / attackFrac : 1f;

        static float EaseOut(float t) => 1f - (1f - t) * (1f - t);

        AudioClip ToClip(string name, float[] data)
        {
            var clip = AudioClip.Create(name, data.Length, 1, SampleRate, false);
            clip.SetData(data, 0);
            return clip;
        }
    }
}
