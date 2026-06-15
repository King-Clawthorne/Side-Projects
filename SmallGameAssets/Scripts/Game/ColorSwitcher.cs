using UnityEngine;

namespace SmallGame
{
    public class ColorSwitcher : MonoBehaviour
    {
        public float rotationSpeed = 90f;
        bool consumed;

        void Update()
        {
            transform.Rotate(0f, 0f, rotationSpeed * Time.deltaTime);
        }

        public void Consume()
        {
            if (consumed) return;
            consumed = true;
            Destroy(gameObject);
        }
    }
}
