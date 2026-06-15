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
            if (cameraFollow != null) cameraFollow.Shake(0.05f, 0.08f);
        }

        public void Switch(Vector3 pos, Color tint)
        {
            Spawn(switchPrefab, pos, tint);
            if (cameraFollow != null) cameraFollow.Shake(0.12f, 0.18f);
        }

        public void Powerup(Vector3 pos, Color tint)
        {
            // Reuse the switch burst with the powerup's tint
            Spawn(switchPrefab, pos, tint);
            if (cameraFollow != null) cameraFollow.Shake(0.08f, 0.14f);
        }

        public void Death(Vector3 pos)
        {
            Spawn(deathPrefab, pos, new Color(0.95f, 0.3f, 0.3f));
            if (cameraFollow != null) cameraFollow.Shake(0.55f, 0.45f);
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
