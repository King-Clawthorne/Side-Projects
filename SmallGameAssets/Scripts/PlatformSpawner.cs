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

        [Header("Power-up respawning (anti-stuck)")]
        [Tooltip("Top up power-ups in the play area so the player can't get stuck for lack of them after falling.")]
        public bool respawnPowerups = true;
        [Tooltip("How often (seconds) to check the play area and top up power-ups.")]
        public float respawnCheckInterval = 1.5f;
        [Tooltip("Target (and maximum) power-up density: power-ups per vertical world-unit of the play area. Top-up never exceeds this.")]
        public float targetPowerupDensity = 0.04f;
        [Tooltip("Extra distance below the camera to include when topping up, so power-ups exist where the player falls back to.")]
        public float respawnRegionMargin = 6f;
        [Tooltip("Minimum vertical gap between a respawned power-up and any existing power-up.")]
        public float respawnMinGap = 2.5f;

        [Header("Color variety")]
        [Tooltip("How many of the most recent platforms count as a 'section'.")]
        public int colorWindow = 6;
        [Tooltip("Each section must contain at least this many distinct colors.")]
        public int minDistinctColors = 3;

        float nextSpawnY;
        readonly List<GameObject> live = new List<GameObject>();

        // Power-up respawn scratch state (reused to avoid per-check allocations).
        float respawnTimer;
        readonly List<GameObject> candidatePlatforms = new List<GameObject>();
        readonly List<float> powerupYs = new List<float>();

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

            if (respawnPowerups)
            {
                respawnTimer -= Time.deltaTime;
                if (respawnTimer <= 0f)
                {
                    respawnTimer = Mathf.Max(0.1f, respawnCheckInterval);
                    TopUpPowerups();
                }
            }
        }

        // Keeps the visible play area (plus a margin below the camera) stocked with
        // power-ups up to a target density, so a player who has fallen back into an
        // already-cleared region still has a way out instead of getting stuck.
        void TopUpPowerups()
        {
            if (cam == null) return;

            float halfH = cam.orthographicSize;
            float camY = cam.transform.position.y;
            float bottom = camY - halfH - respawnRegionMargin;
            float top = camY + halfH;
            float regionHeight = top - bottom;
            if (regionHeight <= 0f) return;

            // Target count is capped by the editor-tunable density; never overshoot it.
            int target = Mathf.FloorToInt(targetPowerupDensity * regionHeight);
            if (target <= 0) return;

            candidatePlatforms.Clear();
            powerupYs.Clear();
            int existing = 0;
            for (int i = 0; i < live.Count; i++)
            {
                var go = live[i];
                if (go == null) continue;
                float y = go.transform.position.y;
                if (y < bottom || y > top) continue;

                if (go.GetComponent<PowerupPickup>() != null)
                {
                    existing++;
                    powerupYs.Add(y);
                }
                else if (go.GetComponent<Platform>() != null)
                {
                    candidatePlatforms.Add(go);
                }
            }

            int toSpawn = target - existing;
            if (toSpawn <= 0 || candidatePlatforms.Count == 0) return;

            float halfW = cam.orthographicSize * cam.aspect - 1.2f;
            int spawned = 0;
            // Each platform is tried at most once (removed on use or rejection), so
            // this terminates even when every candidate is too close to a power-up.
            while (spawned < toSpawn && candidatePlatforms.Count > 0)
            {
                int idx = Random.Range(0, candidatePlatforms.Count);
                var platGo = candidatePlatforms[idx];
                candidatePlatforms.RemoveAt(idx);
                if (platGo == null) continue;

                Vector3 pp = platGo.transform.position;
                float sy = pp.y + Random.Range(0.7f, 1.1f);

                bool tooClose = false;
                for (int k = 0; k < powerupYs.Count; k++)
                {
                    if (Mathf.Abs(powerupYs[k] - sy) < respawnMinGap) { tooClose = true; break; }
                }
                if (tooClose) continue;

                var prefab = PickPowerupPrefab();
                if (prefab == null) return;
                float sx = Mathf.Clamp(pp.x + Random.Range(-1.2f, 1.2f), -halfW, halfW);
                var pu = Instantiate(prefab, new Vector3(sx, sy, 0f), Quaternion.identity);
                pu.transform.parent = transform;
                live.Add(pu);
                powerupYs.Add(sy);
                spawned++;
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
                // Force a new color for the platform above the switcher; the switcher
                // itself adopts the nearest platform's color at runtime (see ColorSwitcher).
                assumedColor = Palette.RandomOther(assumedColor);
                s.GetComponent<ColorSwitcher>().Init(assumedColor);
                forceMatchNext = true;
                sinceMatch = 0;
                spawnedSwitcher = true;
            }

            // Coin gets first pick of the bonus slot above the platform; otherwise a
            // 100% power-up chance would always win the slot and the coin would never
            // appear. Coin and power-ups stay mutually exclusive (same position).
            bool spawnedSomething = spawnedSwitcher;
            if (!spawnedSomething && match && multiplierCoinPrefab != null && Random.value < multiplierCoinChance)
            {
                float sx = Mathf.Clamp(x + Random.Range(-1.2f, 1.2f), -halfW, halfW);
                float sy = y + Random.Range(0.7f, 1.1f);
                var coin = Instantiate(multiplierCoinPrefab, new Vector3(sx, sy, 0f), Quaternion.identity);
                coin.transform.parent = transform;
                live.Add(coin);
                spawnedSomething = true;
            }

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
