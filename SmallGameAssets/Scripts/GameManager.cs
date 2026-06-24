using UnityEngine;
using UnityEngine.SceneManagement;

namespace SmallGame
{
    public class GameManager : MonoBehaviour
    {
        public static GameManager Instance { get; private set; }

        public Transform player;
        public Camera cam;

        public bool IsGameOver { get; private set; }
        public int Score { get; private set; }
        public int BestScore { get; private set; }
        public bool IsNewRecord { get; private set; }

        public float Multiplier { get; private set; } = 1f;
        public float MultiplierTimeLeft { get; private set; }

        float startY;
        float maxHeight;     // raw height climbed (without multiplier)
        float bonusPoints;   // extra points accumulated while a multiplier was active

        void Awake()
        {
            if (Instance != null && Instance != this) { Destroy(gameObject); return; }
            Instance = this;
        }

        void OnDestroy()
        {
            if (Instance == this) Instance = null;
        }

        void Start()
        {
            if (player == null)
            {
                var pgo = GameObject.FindGameObjectWithTag("Player");
                if (pgo != null) player = pgo.transform;
            }
            if (cam == null) cam = Camera.main;
            if (player != null) startY = player.position.y;
            BestScore = HighScore.Get();
        }

        void Update()
        {
            if (IsGameOver) return;
            if (player == null) return;

            float h = Mathf.Max(0f, player.position.y - startY);
            if (h > maxHeight)
            {
                float delta = h - maxHeight;
                maxHeight = h;
                if (Multiplier > 1f) bonusPoints += delta * (Multiplier - 1f);
            }
            Score = Mathf.FloorToInt(maxHeight + bonusPoints);

            if (MultiplierTimeLeft > 0f)
            {
                MultiplierTimeLeft -= Time.deltaTime;
                if (MultiplierTimeLeft <= 0f) { Multiplier = 1f; MultiplierTimeLeft = 0f; }
            }

            if (cam != null && player.position.y < cam.transform.position.y - cam.orthographicSize - 1.5f)
            {
                GameOver();
            }
        }

        public void GrantMultiplier(float multiplier, float duration)
        {
            // Stack: multipliers compound and their durations add together.
            Multiplier *= multiplier;
            MultiplierTimeLeft += duration;
        }

        public void GameOver()
        {
            if (IsGameOver) return;
            IsGameOver = true;
            IsNewRecord = HighScore.TrySet(Score);
            if (IsNewRecord) BestScore = Score;
            if (EffectsManager.Instance != null && player != null)
                EffectsManager.Instance.Death(player.position);
        }

        public void Restart()
        {
            IsGameOver = false;
            SceneManager.LoadScene(SceneManager.GetActiveScene().buildIndex);
        }
    }
}
