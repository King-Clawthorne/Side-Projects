using UnityEngine;

namespace SmallGame
{
    [ExecuteAlways]
    public class Platform : MonoBehaviour
    {
        [SerializeField] ColorId color = ColorId.Red;
        public ColorId Color => color;

        SpriteRenderer sr;

        void Awake() { Apply(); }
        void OnValidate() { Apply(); }

        public void SetColor(ColorId c) { color = c; Apply(); }

        void Apply()
        {
            if (sr == null) sr = GetComponent<SpriteRenderer>();
            if (sr != null) sr.color = Palette.Get(color);
        }
    }
}
