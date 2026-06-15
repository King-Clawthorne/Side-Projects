using UnityEngine;
using UnityEngine.UI;

namespace SmallGame
{
    public class UIController : MonoBehaviour
    {
        public Text scoreText;
        public Text bestText;
        public Text multiplierText;
        public GameObject gameOverPanel;
        public Text finalScoreText;
        public GameObject newRecordLabel;
        public Button restartButton;

        void Start()
        {
            if (gameOverPanel != null) gameOverPanel.SetActive(false);
            if (newRecordLabel != null) newRecordLabel.SetActive(false);
            if (multiplierText != null) multiplierText.gameObject.SetActive(false);
            if (restartButton != null)
            {
                restartButton.onClick.RemoveAllListeners();
                restartButton.onClick.AddListener(OnRestart);
            }
        }

        void Update()
        {
            var gm = GameManager.Instance;
            if (gm == null) return;
            if (scoreText != null) scoreText.text = "Score: " + gm.Score;
            if (bestText != null) bestText.text = "Best: " + gm.BestScore;

            if (multiplierText != null)
            {
                bool active = gm.Multiplier > 1f && gm.MultiplierTimeLeft > 0f;
                multiplierText.gameObject.SetActive(active);
                if (active)
                    multiplierText.text = "x" + gm.Multiplier.ToString("0.#") + "  " + gm.MultiplierTimeLeft.ToString("0.0") + "s";
            }

            if (gameOverPanel != null && gm.IsGameOver && !gameOverPanel.activeSelf)
            {
                gameOverPanel.SetActive(true);
                if (finalScoreText != null)
                    finalScoreText.text = "Score: " + gm.Score + "   Best: " + gm.BestScore;
                if (newRecordLabel != null) newRecordLabel.SetActive(gm.IsNewRecord);
            }
        }

        public void OnRestart()
        {
            if (GameManager.Instance != null) GameManager.Instance.Restart();
        }
    }
}
