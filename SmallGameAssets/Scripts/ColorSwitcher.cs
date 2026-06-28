using UnityEngine;

namespace SmallGame
{
    public class ColorSwitcher : MonoBehaviour
    {
        public float rotationSpeed = 90f;
        public float normalScale = 0.35f;
        public float targetScale = 0.55f;
        public ColorId targetColor;
        bool consumed;
        bool colorLocked;

        public void Init(ColorId target)
        {
            targetColor = target;
            ApplyHighlight();
        }

        void Update()
        {
            // The spawner can't know the switcher's nearest platform yet (the platform
            // above it spawns later in the same frame), so we resolve it here on the
            // first frame the switcher is alive, once its neighbours exist. The switcher
            // then shows the color of whatever platform is physically closest to it.
            if (!colorLocked) LockToNearestPlatform();
            transform.Rotate(0f, 0f, rotationSpeed * Time.deltaTime);
        }

        void LockToNearestPlatform()
        {
            var platforms = FindObjectsByType<Platform>(FindObjectsInactive.Exclude);
            if (platforms.Length == 0) return; // neighbours not spawned yet; try next frame

            Platform nearest = null;
            float best = float.MaxValue;
            Vector2 pos = transform.position;
            for (int i = 0; i < platforms.Length; i++)
            {
                float d = ((Vector2)platforms[i].transform.position - pos).sqrMagnitude;
                if (d < best) { best = d; nearest = platforms[i]; }
            }
            if (nearest != null)
            {
                targetColor = nearest.Color;
                ApplyHighlight();
                colorLocked = true;
            }
        }

        // Enlarge the quadrant matching targetColor so the switcher reads as that color.
        void ApplyHighlight()
        {
            for (int i = 0; i < transform.childCount; i++)
            {
                bool isTarget = i == (int)targetColor;
                float s = isTarget ? targetScale : normalScale;
                transform.GetChild(i).localScale = new Vector3(s, s, 1f);
            }
        }

        public void Consume()
        {
            if (consumed) return;
            consumed = true;
            Destroy(gameObject);
        }
    }
}
