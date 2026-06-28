using UnityEngine;

namespace SmallGame
{
    public class EffectsManager : MonoBehaviour
    {
        public static EffectsManager Instance { get; private set; }

        public GameObject bouncePrefab;
        public GameObject switchPrefab;
        public GameObject deathPrefab;
        public CameraFollow cameraFollow;

        [System.Serializable]
        public struct ShakeSettings
        {
            [Tooltip("How far the camera kicks.")] public float amplitude;
            [Tooltip("How long the shake lasts, in seconds.")] public float duration;

            public ShakeSettings(float amplitude, float duration)
            {
                this.amplitude = amplitude;
                this.duration = duration;
            }
        }

        [Header("Camera shake per event")]
        public ShakeSettings bounceShake = new ShakeSettings(0.05f, 0.08f);
        public ShakeSettings switchShake = new ShakeSettings(0.12f, 0.18f);
        public ShakeSettings powerupShake = new ShakeSettings(0.08f, 0.14f);
        public ShakeSettings deathShake = new ShakeSettings(0.55f, 0.45f);

        void Awake()
        {
            if (Instance != null && Instance != this) { Destroy(gameObject); return; }
            Instance = this;
        }

        void OnDestroy()
        {
            if (Instance == this) Instance = null;
        }

        void Start()
        {
            if (cameraFollow == null && Camera.main != null)
                cameraFollow = Camera.main.GetComponent<CameraFollow>();
        }

        public void Bounce(Vector3 pos, Color tint)
        {
            Spawn(bouncePrefab, pos, tint);
            if (cameraFollow != null) cameraFollow.Shake(bounceShake.amplitude, bounceShake.duration);
            SfxManager.Instance.PlayBounce();
        }

        public void Switch(Vector3 pos, Color tint)
        {
            Spawn(switchPrefab, pos, tint);
            if (cameraFollow != null) cameraFollow.Shake(switchShake.amplitude, switchShake.duration);
            SfxManager.Instance.PlaySwitch();
        }

        public void Powerup(Vector3 pos, Color tint)
        {
            // Reuse the switch burst with the powerup's tint
            Spawn(switchPrefab, pos, tint);
            if (cameraFollow != null) cameraFollow.Shake(powerupShake.amplitude, powerupShake.duration);
            SfxManager.Instance.PlayPowerup();
        }

        public void Death(Vector3 pos)
        {
            Spawn(deathPrefab, pos, new Color(0.95f, 0.3f, 0.3f));
            if (cameraFollow != null) cameraFollow.Shake(deathShake.amplitude, deathShake.duration);
            SfxManager.Instance.PlayDeath();
        }

        void Spawn(GameObject prefab, Vector3 pos, Color tint)
        {
            if (prefab == null) return;
            var go = Instantiate(prefab, pos, Quaternion.identity);
            var ps = go.GetComponent<ParticleSystem>();
            if (ps != null)
            {
                var main = ps.main;
                main.startColor = tint;
                ps.Play();
            }
        }
    }
}
