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

        public void Init(ColorId target)
        {
            targetColor = target;
            for (int i = 0; i < transform.childCount; i++)
            {
                bool isTarget = i == (int)target;
                float s = isTarget ? targetScale : normalScale;
                transform.GetChild(i).localScale = new Vector3(s, s, 1f);
            }
        }

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
