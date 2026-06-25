using UnityEngine;

namespace SmallGame
{
    public class CameraFollow : MonoBehaviour
    {
        public Transform target;
        public float yOffset = 1.5f;
        public float smooth = 5f;

        float baseX;
        float baseY;
        float shakeTimer;
        float shakeDuration;
        float shakeAmplitude;

        void Start()
        {
            baseX = transform.position.x;
            baseY = transform.position.y;
        }

        public void Shake(float amplitude, float duration)
        {
            if (amplitude > shakeAmplitude || shakeTimer <= 0f)
            {
                shakeAmplitude = amplitude;
                shakeDuration = Mathf.Max(0.0001f, duration);
                shakeTimer = shakeDuration;
            }
        }

        void LateUpdate()
        {
            if (target != null)
            {
                float desired = target.position.y + yOffset;
                baseY = Mathf.Lerp(baseY, desired, Time.deltaTime * smooth);
            }

            float ox = 0f, oy = 0f;
            if (shakeTimer > 0f)
            {
                shakeTimer -= Time.deltaTime;
                float t = Mathf.Clamp01(shakeTimer / shakeDuration);
                Vector2 r = Random.insideUnitCircle * shakeAmplitude * t;
                ox = r.x; oy = r.y;
            }
            var p = transform.position;
            p.x = baseX + ox;
            p.y = baseY + oy;
            transform.position = p;
        }
    }
}
