using UnityEngine;

namespace SmallGame
{
    // Add this alongside a Platform component to make it apply a horizontal
    // kick (and a stronger vertical bounce) when the player lands on it.
    public class RocketPlatform : MonoBehaviour
    {
        public float horizontalKick = 12f;
        public float bounceMultiplier = 1.25f;
        // +1 = right, -1 = left. Set at spawn time.
        public int direction = 1;

        public Transform arrowVisual; // optional child arrow (whole arrow group)

        void Start() { ApplyVisualDirection(); }

        public void SetDirection(int dir)
        {
            direction = dir >= 0 ? 1 : -1;
            ApplyVisualDirection();
        }

        void ApplyVisualDirection()
        {
            // Heal stale/missing references (e.g. prefab saved before the field
            // type changed) by falling back to the child named "Arrow".
            if (!arrowVisual)
            {
                var t = transform.Find("Arrow");
                arrowVisual = t != null ? t : null;
            }
            if (arrowVisual != null)
            {
                var s = arrowVisual.localScale;
                s.x = Mathf.Abs(s.x) * direction;
                arrowVisual.localScale = s;
            }
        }
    }
}
