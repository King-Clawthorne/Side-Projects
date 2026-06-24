using UnityEngine;

namespace SmallGame
{
    public class MultiplierCoin : MonoBehaviour
    {
        public float multiplier = 2f;
        public float duration = 8f;
        public float bobAmplitude = 0.15f;
        public float bobSpeed = 2.5f;

        Vector3 basePos;
        bool consumed;

        void Start() { basePos = transform.position; }

        void Update()
        {
            transform.position = basePos + new Vector3(0f, Mathf.Sin(Time.time * bobSpeed) * bobAmplitude, 0f);
            transform.Rotate(0f, 180f * Time.deltaTime, 0f);
        }

        void OnTriggerEnter2D(Collider2D other)
        {
            if (consumed) return;
            var pc = other.GetComponent<PlayerController>();
            if (pc == null) pc = other.GetComponentInParent<PlayerController>();
            if (pc == null) return;

            consumed = true;
            if (GameManager.Instance != null)
                GameManager.Instance.GrantMultiplier(multiplier, duration);
            if (EffectsManager.Instance != null)
                EffectsManager.Instance.Powerup(transform.position, new Color(1f, 0.85f, 0.2f));
            Destroy(gameObject);
        }
    }
}
