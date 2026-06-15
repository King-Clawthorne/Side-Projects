using UnityEngine;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

namespace SmallGame
{
    [RequireComponent(typeof(Rigidbody2D))]
    [RequireComponent(typeof(SpriteRenderer))]
    public class PlayerController : MonoBehaviour
    {
        public float horizontalSpeed = 6f;
        public float bounceVelocity = 14f;
        public ColorId currentColor = ColorId.Red;

        public SpriteRenderer shieldVisual;     // optional child halo
        public SpriteRenderer jetpackVisual;    // optional child flame

        Rigidbody2D rb;
        SpriteRenderer sr;
        Camera cam;

        // Power-up state
        bool hasShield;
        float jetpackTimer;
        float jetpackSpeed;
        float bounceMultiplier = 1f;

        public bool HasShield => hasShield;
        public bool JetpackActive => jetpackTimer > 0f;

        void Awake()
        {
            rb = GetComponent<Rigidbody2D>();
            sr = GetComponent<SpriteRenderer>();
            cam = Camera.main;
            ApplyColor();
            UpdateVisuals();
        }

        void Update()
        {
            if (GameManager.Instance != null && GameManager.Instance.IsGameOver)
            {
                var dv = rb.linearVelocity;
                dv.x = 0f;
                rb.linearVelocity = dv;
                return;
            }

            float x = 0f;
#if ENABLE_INPUT_SYSTEM
            var kb = Keyboard.current;
            if (kb != null)
            {
                if (kb.aKey.isPressed || kb.leftArrowKey.isPressed) x -= 1f;
                if (kb.dKey.isPressed || kb.rightArrowKey.isPressed) x += 1f;
            }
#else
            x = Input.GetAxisRaw("Horizontal");
#endif
            var v = rb.linearVelocity;
            v.x = x * horizontalSpeed;

            if (jetpackTimer > 0f)
            {
                jetpackTimer -= Time.deltaTime;
                v.y = jetpackSpeed;
                if (jetpackTimer <= 0f) UpdateVisuals();
            }
            rb.linearVelocity = v;

            if (cam != null)
            {
                float halfH = cam.orthographicSize;
                float halfW = halfH * cam.aspect;
                Vector2 p = rb.position;
                float camX = cam.transform.position.x;
                if (p.x > camX + halfW + 0.3f) { p.x = camX - halfW - 0.2f; rb.position = p; }
                else if (p.x < camX - halfW - 0.3f) { p.x = camX + halfW + 0.2f; rb.position = p; }
            }
        }

        public void SetColor(ColorId c)
        {
            currentColor = c;
            ApplyColor();
        }

        void ApplyColor()
        {
            if (sr != null) sr.color = Palette.Get(currentColor);
        }

        public void GrantShield()
        {
            hasShield = true;
            UpdateVisuals();
        }

        public void QueueBounceBoost(float multiplier)
        {
            bounceMultiplier = Mathf.Max(bounceMultiplier, multiplier);
        }

        public void ActivateJetpack(float duration, float speed)
        {
            jetpackTimer = Mathf.Max(jetpackTimer, duration);
            jetpackSpeed = speed;
            UpdateVisuals();
        }

        void UpdateVisuals()
        {
            if (shieldVisual != null) shieldVisual.enabled = hasShield;
            if (jetpackVisual != null) jetpackVisual.enabled = jetpackTimer > 0f;
        }

        void OnCollisionEnter2D(Collision2D collision)
        {
            var platform = collision.collider.GetComponent<Platform>();
            if (platform == null) platform = collision.collider.GetComponentInParent<Platform>();
            if (platform == null) return;

            bool fromAbove = false;
            Vector3 contactPoint = transform.position;
            for (int i = 0; i < collision.contactCount; i++)
            {
                var c = collision.GetContact(i);
                if (c.normal.y > 0.5f) { fromAbove = true; contactPoint = c.point; break; }
            }
            if (!fromAbove) return;

            // Jetpack ignores landings (passes through)
            if (jetpackTimer > 0f) return;

            bool match = platform.Color == currentColor;
            if (!match && hasShield)
            {
                hasShield = false;
                UpdateVisuals();
                match = true; // consume shield, treat as a successful bounce
                if (EffectsManager.Instance != null)
                    EffectsManager.Instance.Powerup(contactPoint, new Color(0.55f, 0.95f, 1f));
            }

            if (match)
            {
                var rocket = collision.collider.GetComponent<RocketPlatform>();
                if (rocket == null) rocket = collision.collider.GetComponentInParent<RocketPlatform>();

                var v = rb.linearVelocity;
                float vMul = bounceMultiplier * (rocket != null ? rocket.bounceMultiplier : 1f);
                v.y = bounceVelocity * vMul;
                if (rocket != null) v.x += rocket.horizontalKick * rocket.direction;
                rb.linearVelocity = v;
                if (EffectsManager.Instance != null)
                    EffectsManager.Instance.Bounce(contactPoint, Palette.Get(currentColor));
                bounceMultiplier = 1f;
            }
            else
            {
                if (GameManager.Instance != null) GameManager.Instance.GameOver();
            }
        }

        void OnTriggerEnter2D(Collider2D other)
        {
            var sw = other.GetComponent<ColorSwitcher>();
            if (sw == null) sw = other.GetComponentInParent<ColorSwitcher>();
            if (sw != null)
            {
                Vector3 swPos = sw.transform.position;
                SetColor(Palette.RandomOther(currentColor));
                if (EffectsManager.Instance != null)
                    EffectsManager.Instance.Switch(swPos, Palette.Get(currentColor));
                sw.Consume();
                return;
            }

            var pickup = other.GetComponent<PowerupPickup>();
            if (pickup == null) pickup = other.GetComponentInParent<PowerupPickup>();
            if (pickup != null)
            {
                pickup.Consume(this);
            }
        }
    }
}
