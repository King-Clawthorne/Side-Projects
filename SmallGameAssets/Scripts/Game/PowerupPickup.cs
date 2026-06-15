using UnityEngine;

namespace SmallGame
{
    public enum PowerupKind { Spring, Shield, Jetpack }

    public abstract class PowerupPickup : MonoBehaviour
    {
        public float bobAmplitude = 0.15f;
        public float bobSpeed = 2f;
        Vector3 basePos;
        bool consumed;

        public abstract PowerupKind Kind { get; }
        public abstract Color Tint { get; }

        protected virtual void Start() { basePos = transform.position; }

        protected virtual void Update()
        {
            transform.position = basePos + new Vector3(0f, Mathf.Sin(Time.time * bobSpeed) * bobAmplitude, 0f);
            transform.Rotate(0f, 0f, 60f * Time.deltaTime);
        }

        public void Consume(PlayerController player)
        {
            if (consumed) return;
            consumed = true;
            Apply(player);
            if (EffectsManager.Instance != null)
                EffectsManager.Instance.Powerup(transform.position, Tint);
            Destroy(gameObject);
        }

        protected abstract void Apply(PlayerController player);
    }
}
