using UnityEngine;

namespace SmallGame
{
    public static class HighScore
    {
        const string Key = "SmallGame.HighScore";
        public static int Get() => PlayerPrefs.GetInt(Key, 0);
        public static bool TrySet(int score)
        {
            if (score > Get())
            {
                PlayerPrefs.SetInt(Key, score);
                PlayerPrefs.Save();
                return true;
            }
            return false;
        }
    }
}
