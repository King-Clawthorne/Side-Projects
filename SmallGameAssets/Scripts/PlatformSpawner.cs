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
        public int maxDifficultyScore = 200;

        [Header("Color variety")]
        [Tooltip("How many of the most recent platforms count as a 'section'.")]
        public int colorWindow = 6;
        [Tooltip("Each section must contain at least this many distinct colors.")]
        public int minDistinctColors = 3;

        float nextSpawnY;
        readonly List<GameObject> live = new List<GameObject>();

        // Reachability state
        ColorId assumedColor;
        int sinceMatch;
        bool forceMatchNext;

        // Colors of the most recently spawned platforms (sliding window).
        readonly Queue<ColorId> recentColors = new Queue<ColorId>();

        void Start()
        {
            if (cam == null) cam = Camera.main;
            nextSpawnY = (player != null ? player.position.y : 0f) + 2f;
            var pc = player != null ? player.GetComponent<PlayerController>() : null;
            assumedColor = pc != null ? pc.currentColor : ColorId.Red;
            sinceMatch = 0;
            forceMatchNext = false;
            recentColors.Clear();
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

            for (int i = live.Count - 1; i >= 0; i--)
            {
                if (live[i] == null) live.RemoveAt(i);
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
            // Reachability comes first: there must always be a landable platform
            // within one step, so a forced match overrides everything else.
            bool mustMatch = forceMatchNext || sinceMatch >= 1;
            // If the recent section has collapsed to too few colors, inject an
            // off-color platform to widen the palette (unless we must match).
            bool needVariety = recentColors.Count >= colorWindow
                && DistinctRecentColors() < minDistinctColors;
            bool match = mustMatch || (!needVariety && Random.value < CurrentMatchProb());
            if (match)
            {
                color = assumedColor;
                sinceMatch = 0;
                forceMatchNext = false;
            }
            else
            {
                color = PickDiverseOther(assumedColor);
                sinceMatch++;
            }
            RecordColor(color);

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
                s.GetComponent<ColorSwitcher>().Init(assumedColor);
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

        void RecordColor(ColorId c)
        {
            recentColors.Enqueue(c);
            while (recentColors.Count > colorWindow) recentColors.Dequeue();
        }

        int DistinctRecentColors()
        {
            int mask = 0;
            foreach (var c in recentColors) mask |= 1 << (int)c;
            int count = 0;
            while (mask != 0) { count += mask & 1; mask >>= 1; }
            return count;
        }

        // An off-color platform color, preferring one absent from the recent
        // window so the section keeps a healthy spread of colors.
        ColorId PickDiverseOther(ColorId not)
        {
            ColorId pick = not;
            int seen = 0;
            for (int i = 0; i < Palette.Colors.Length; i++)
            {
                var c = (ColorId)i;
                if (c == not || recentColors.Contains(c)) continue;
                // Reservoir pick so every fresh candidate is equally likely.
                seen++;
                if (Random.Range(0, seen) == 0) pick = c;
            }
            return seen > 0 ? pick : Palette.RandomOther(not);
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
