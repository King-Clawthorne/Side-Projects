using System.Collections.Generic;
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
        Collider2D playerCollider;

        // Platforms we're currently passing through because their color doesn't match.
        readonly List<Collider2D> ignoredColliders = new List<Collider2D>();

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
            playerCollider = GetComponent<Collider2D>();
            cam = Camera.main;
            ApplyColor();
            UpdateVisuals();
        }

        void Update()
        {
            if (GameManager.Instance != null && GameManager.Instance.IsGameOver)
            {
                // Fully stop the player once dead: halt motion and freeze the body.
                if (rb.bodyType != RigidbodyType2D.Static)
                {
                    rb.linearVelocity = Vector2.zero;
                    rb.angularVelocity = 0f;
                    rb.bodyType = RigidbodyType2D.Static;
                }
                return;
            }

            // Game is active: make sure the body can move (e.g. after a restart).
            if (rb.bodyType == RigidbodyType2D.Static)
                rb.bodyType = RigidbodyType2D.Dynamic;

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
            // Re-enable collisions with platforms we were passing through; they may
            // now match the new color and should be landable again.
            ClearIgnoredCollisions();
        }

        // Color of the platform nearest the player. Falls back to the supplied
        // default when there are no platforms in the scene.
        ColorId NearestPlatformColor(ColorId fallback)
        {
            var platforms = FindObjectsByType<Platform>(FindObjectsInactive.Exclude);
            Platform nearest = null;
            float best = float.MaxValue;
            Vector2 pos = rb.position;
            for (int i = 0; i < platforms.Length; i++)
            {
                float d = ((Vector2)platforms[i].transform.position - pos).sqrMagnitude;
                if (d < best) { best = d; nearest = platforms[i]; }
            }
            return nearest != null ? nearest.Color : fallback;
        }

        void ClearIgnoredCollisions()
        {
            if (playerCollider != null)
            {
                for (int i = 0; i < ignoredColliders.Count; i++)
                {
                    if (ignoredColliders[i] != null)
                        Physics2D.IgnoreCollision(playerCollider, ignoredColliders[i], false);
                }
            }
            ignoredColliders.Clear();
        }

        void ApplyColor()
        {
            if (sr != null) sr.color = Palette.Get(currentColor);
        }

        public void GrantShield()
        {
            hasShield = true;
            ClearIgnoredCollisions();
            UpdateVisuals();
        }

        public void QueueBounceBoost(float multiplier)
        {
            // Stack: multiple springs compound the next bounce.
            bounceMultiplier *= multiplier;
        }

        public void ActivateJetpack(float duration, float speed)
        {
            // Stack: durations add, and the strongest speed wins.
            jetpackTimer += duration;
            jetpackSpeed = Mathf.Max(jetpackSpeed, speed);
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

            bool match = platform.Color == currentColor;

            // Different color (and no shield): fall through the platform instead of
            // dying. Ignore this collider so the player passes straight through it.
            if (!match && !hasShield)
            {
                if (playerCollider != null)
                {
                    Physics2D.IgnoreCollision(playerCollider, collision.collider, true);
                    ignoredColliders.Add(collision.collider);
                }
                return;
            }

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

            if (!match && hasShield)
            {
                // Shield lets the player land on an off-color platform, and is
                // consumed when used to do so. Same-color landings don't use it.
                hasShield = false;
                UpdateVisuals();
                match = true;
                if (EffectsManager.Instance != null)
                    EffectsManager.Instance.Powerup(contactPoint, new Color(0.55f, 0.95f, 1f));
            }

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

        void OnTriggerEnter2D(Collider2D other)
        {
            var sw = other.GetComponent<ColorSwitcher>();
            if (sw == null) sw = other.GetComponentInParent<ColorSwitcher>();
            if (sw != null)
            {
                Vector3 swPos = sw.transform.position;
                SetColor(sw.targetColor);
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
