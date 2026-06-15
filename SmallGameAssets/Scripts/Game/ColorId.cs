using UnityEngine;

namespace SmallGame
{
    public enum ColorId { Red = 0, Blue = 1, Yellow = 2, Green = 3 }

    public static class Palette
    {
        public static readonly Color[] Colors = new Color[]
        {
            new Color(0.95f, 0.30f, 0.30f), // Red
            new Color(0.30f, 0.55f, 0.95f), // Blue
            new Color(0.97f, 0.85f, 0.25f), // Yellow
            new Color(0.35f, 0.85f, 0.45f), // Green
        };

        public static Color Get(ColorId id) => Colors[(int)id];

        public static ColorId Random() => (ColorId)UnityEngine.Random.Range(0, Colors.Length);

        public static ColorId RandomOther(ColorId not)
        {
            ColorId c;
            do { c = Random(); } while (c == not);
            return c;
        }
    }
}
