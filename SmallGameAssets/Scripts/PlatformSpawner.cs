using System.Collections.Generic;
using UnityEngine;

namespace SmallGame
{
    public class PlatformSpawner : MonoBehaviour
    {
        public GameObject platformPrefab;
        public GameObject rocketPlatformPrefab;
        public GameObject switcherPrefab;

        [Header("Power-up prefabs")]
        public GameObject springPrefab;
        public GameObject shieldPrefab;
        public GameObject jetpackPrefab;
        public GameObject multiplierCoinPrefab;

        [Header("Rocket platform")]
        [Range(0f, 1f)] public float rocketChance = 0.06f;

        [Header("Multiplier coin")]
        [Range(0f, 1f)] public float multiplierCoinChance = 0.05f;

        public Camera cam;
        public Transform player;

        [Header("Base spacing (easy)")]
        public float minSpacingY = 1.6f;
        public float maxSpacingY = 2.4f;

        [Header("Hard spacing (at maxDifficultyScore)")]
        public float minSpacingYHard = 1.9f;
        public float maxSpacingYHard = 2.7f;

        [Header("Color rules")]
        [Range(0f, 1f)] public float matchProbEasy = 0.85f;
        [Range(0f, 1f)] public float matchProbHard = 0.55f;
        [Range(0f, 1f)] public float switcherChanceEasy = 0.20f;
        [Range(0f, 1f)] public float switcherChanceHard = 0.30f;

        [Header("Power-up chances (per matching platform)")]
        [Range(0f, 1f)] public float powerupChance = 0.10f;
        [Range(0f, 1f)] public float springWeight = 1f;
        [Range(0f, 1f)] public float shieldWeight = 1f;
        [Range(0f, 1f)] public float jetpackWeight = 0.6f;

        [Header("Platform width")]
        public float platformWidthEasy = 1.6f;
        public float platformWidthHard = 1.1f;

        [Header("Spawn / despawn")]
        public float spawnAheadY = 14f;
        public float despawnBelowY = 12f;
        public int maxDifficultyScore = 200;

        float nextSpawnY;
        readonly List<GameObject> live = new List<GameObject>();

        // Reachability state
        ColorId assumedColor;
        int sinceMatch;
        bool forceMatchNext;

        void Start()
        {
            if (cam == null) cam = Camera.main;
            nextSpawnY = (player != null ? player.position.y : 0f) + 2f;
            var pc = player != null ? player.GetComponent<PlayerController>() : null;
            assumedColor = pc != null ? pc.currentColor : ColorId.Red;
            sinceMatch = 0;
            forceMatchNext = false;
        }

        void Update()
        {
            if (cam == null) return;
            float topY = cam.transform.position.y + spawnAheadY;
            while (nextSpawnY < topY)
            {
                SpawnPlatformAt(nextSpawnY);
                nextSpawnY += Random.Range(CurrentMinSpacing(), CurrentMaxSpacing());
            }

            float cutoff = cam.transform.position.y - despawnBelowY;
            for (int i = live.Count - 1; i >= 0; i--)
            {
                var go = live[i];
                if (go == null) { live.RemoveAt(i); continue; }
                if (go.transform.position.y < cutoff)
                {
                    Destroy(go);
                    live.RemoveAt(i);
                }
            }
        }

        float Difficulty()
        {
            int s = (GameManager.Instance != null) ? GameManager.Instance.Score : 0;
            return Mathf.Clamp01((float)s / Mathf.Max(1, maxDifficultyScore));
        }

        float CurrentMinSpacing() => Mathf.Lerp(minSpacingY, minSpacingYHard, Difficulty());
        float CurrentMaxSpacing() => Mathf.Lerp(maxSpacingY, maxSpacingYHard, Difficulty());
        float CurrentMatchProb() => Mathf.Lerp(matchProbEasy, matchProbHard, Difficulty());
        float CurrentSwitcherChance() => Mathf.Lerp(switcherChanceEasy, switcherChanceHard, Difficulty());
        float CurrentPlatformWidth() => Mathf.Lerp(platformWidthEasy, platformWidthHard, Difficulty());

        void SpawnPlatformAt(float y)
        {
            if (platformPrefab == null || cam == null) return;
            float halfW = cam.orthographicSize * cam.aspect - 1.2f;
            float x = Random.Range(-halfW, halfW);

            ColorId color;
            bool match = forceMatchNext || sinceMatch >= 1 || Random.value < CurrentMatchProb();
            if (match)
            {
                color = assumedColor;
                sinceMatch = 0;
                forceMatchNext = false;
            }
            else
            {
                color = Palette.RandomOther(assumedColor);
                sinceMatch++;
            }

            // Use rocket variant occasionally, only on matching platforms
            bool useRocket = match && rocketPlatformPrefab != null && Random.value < rocketChance;
            var prefabToUse = useRocket ? rocketPlatformPrefab : platformPrefab;
            var p = Instantiate(prefabToUse, new Vector3(x, y, 0f), Quaternion.identity);
            p.transform.parent = transform;
            var sc = p.transform.localScale;
            sc.x = CurrentPlatformWidth();
            p.transform.localScale = sc;

            var plat = p.GetComponent<Platform>();
            if (plat != null) plat.SetColor(color);
            if (useRocket)
            {
                var rp = p.GetComponent<RocketPlatform>();
                if (rp != null) rp.SetDirection(Random.value < 0.5f ? -1 : 1);
            }
            live.Add(p);

            // Switcher (mutually exclusive with power-ups, only on matching platforms)
            bool spawnedSwitcher = false;
            if (match && switcherPrefab != null && Random.value < CurrentSwitcherChance())
            {
                float sx = Mathf.Clamp(x + Random.Range(-1.5f, 1.5f), -halfW, halfW);
                float sy = y + Random.Range(0.7f, 1.1f);
                var s = Instantiate(switcherPrefab, new Vector3(sx, sy, 0f), Quaternion.identity);
                s.transform.parent = transform;
                live.Add(s);
                assumedColor = Palette.RandomOther(assumedColor);
                forceMatchNext = true;
                sinceMatch = 0;
                spawnedSwitcher = true;
            }

            bool spawnedSomething = spawnedSwitcher;
            if (!spawnedSomething && match && Random.value < powerupChance)
            {
                var prefab = PickPowerupPrefab();
                if (prefab != null)
                {
                    float sx = Mathf.Clamp(x + Random.Range(-1.2f, 1.2f), -halfW, halfW);
                    float sy = y + Random.Range(0.7f, 1.1f);
                    var pu = Instantiate(prefab, new Vector3(sx, sy, 0f), Quaternion.identity);
                    pu.transform.parent = transform;
                    live.Add(pu);
                    spawnedSomething = true;
                }
            }

            if (!spawnedSomething && match && multiplierCoinPrefab != null && Random.value < multiplierCoinChance)
            {
                float sx = Mathf.Clamp(x + Random.Range(-1.2f, 1.2f), -halfW, halfW);
                float sy = y + Random.Range(0.7f, 1.1f);
                var coin = Instantiate(multiplierCoinPrefab, new Vector3(sx, sy, 0f), Quaternion.identity);
                coin.transform.parent = transform;
                live.Add(coin);
            }
        }

        GameObject PickPowerupPrefab()
        {
            float ws = (springPrefab != null ? springWeight : 0f);
            float wh = (shieldPrefab != null ? shieldWeight : 0f);
            float wj = (jetpackPrefab != null ? jetpackWeight : 0f);
            float total = ws + wh + wj;
            if (total <= 0f) return null;
            float r = Random.value * total;
            if (r < ws) return springPrefab;
            r -= ws;
            if (r < wh) return shieldPrefab;
            return jetpackPrefab;
        }
    }
}
