#if UNITY_EDITOR
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
using UnityEngine.EventSystems;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem.UI;
#endif
using SmallGame;

public static class SceneBuilder
{
    [MenuItem("SmallGame/Build Doodle Color Scene")]
    public static void Build()
    {
        var squareSprite = AssetDatabase.GetBuiltinExtraResource<Sprite>("UI/Skin/Background.psd");
        if (squareSprite == null) squareSprite = AssetDatabase.GetBuiltinExtraResource<Sprite>("UI/Skin/UISprite.psd");
        var circleSprite = AssetDatabase.GetBuiltinExtraResource<Sprite>("UI/Skin/Knob.psd");
        if (circleSprite == null) circleSprite = GenerateCircleSprite();

        var cam = Camera.main;
        if (cam == null)
        {
            var camGO = new GameObject("Main Camera", typeof(Camera));
            camGO.tag = "MainCamera";
            cam = camGO.GetComponent<Camera>();
        }
        cam.orthographic = true;
        cam.orthographicSize = 6f;
        cam.backgroundColor = new Color(0.12f, 0.12f, 0.16f);
        cam.transform.position = new Vector3(0f, 0f, -10f);
        var cf = cam.GetComponent<CameraFollow>();
        if (cf == null) cf = cam.gameObject.AddComponent<CameraFollow>();

        // Player
        var existingPlayer = GameObject.Find("Player");
        if (existingPlayer != null) Object.DestroyImmediate(existingPlayer);
        var player = new GameObject("Player");
        player.tag = "Player";
        player.transform.position = new Vector3(0f, -2f, 0f);
        player.transform.localScale = new Vector3(0.6f, 0.6f, 1f);
        var psr = player.AddComponent<SpriteRenderer>();
        psr.sprite = circleSprite;
        psr.sortingOrder = 10;
        var rb = player.AddComponent<Rigidbody2D>();
        rb.gravityScale = 3f;
        rb.constraints = RigidbodyConstraints2D.FreezeRotation;
        rb.collisionDetectionMode = CollisionDetectionMode2D.Continuous;
        var pcc = player.AddComponent<CircleCollider2D>();
        pcc.radius = 0.45f;
        var pcomp = player.AddComponent<PlayerController>();

        // Shield halo (cyan ring) — disabled by default
        var shieldGO = new GameObject("ShieldHalo");
        shieldGO.transform.SetParent(player.transform, false);
        shieldGO.transform.localScale = new Vector3(1.6f, 1.6f, 1f);
        var shieldSR = shieldGO.AddComponent<SpriteRenderer>();
        shieldSR.sprite = circleSprite;
        shieldSR.color = new Color(0.55f, 0.95f, 1f, 0.45f);
        shieldSR.sortingOrder = 9;
        shieldSR.enabled = false;
        pcomp.shieldVisual = shieldSR;

        // Jetpack flame (orange streak below) — disabled by default
        var jetGO = new GameObject("JetpackFlame");
        jetGO.transform.SetParent(player.transform, false);
        jetGO.transform.localPosition = new Vector3(0f, -0.55f, 0f);
        jetGO.transform.localScale = new Vector3(0.5f, 1.2f, 1f);
        var jetSR = jetGO.AddComponent<SpriteRenderer>();
        jetSR.sprite = squareSprite;
        jetSR.color = new Color(1f, 0.6f, 0.2f, 0.95f);
        jetSR.sortingOrder = 9;
        jetSR.enabled = false;
        pcomp.jetpackVisual = jetSR;

        // Platform Prefab
        Directory.CreateDirectory("Assets/Prefabs");
        var platTemp = new GameObject("PlatformPrefabTemp");
        platTemp.transform.localScale = new Vector3(1.6f, 0.3f, 1f);
        var plsr = platTemp.AddComponent<SpriteRenderer>();
        plsr.sprite = squareSprite;
        plsr.sortingOrder = 5;
        var pbox = platTemp.AddComponent<BoxCollider2D>();
        pbox.usedByEffector = true;
        var peff = platTemp.AddComponent<PlatformEffector2D>();
        peff.useOneWay = true;
        peff.surfaceArc = 170f;
        platTemp.AddComponent<SmallGame.Platform>();
        var platformPrefab = PrefabUtility.SaveAsPrefabAsset(platTemp, "Assets/Prefabs/PlatformPrefab.prefab");
        Object.DestroyImmediate(platTemp);

        // ColorSwitcher Prefab
        var swTemp = new GameObject("ColorSwitcherPrefabTemp");
        var swCol = swTemp.AddComponent<CircleCollider2D>();
        swCol.isTrigger = true;
        swCol.radius = 0.45f;
        swTemp.AddComponent<ColorSwitcher>();
        for (int i = 0; i < 4; i++)
        {
            var q = new GameObject("Q" + i);
            q.transform.SetParent(swTemp.transform, false);
            float ang = i * 90f * Mathf.Deg2Rad;
            q.transform.localPosition = new Vector3(Mathf.Cos(ang) * 0.35f, Mathf.Sin(ang) * 0.35f, 0f);
            q.transform.localScale = new Vector3(0.35f, 0.35f, 1f);
            var qsr = q.AddComponent<SpriteRenderer>();
            qsr.sprite = squareSprite;
            qsr.color = Palette.Get((ColorId)i);
            qsr.sortingOrder = 6;
        }
        var switcherPrefab = PrefabUtility.SaveAsPrefabAsset(swTemp, "Assets/Prefabs/ColorSwitcherPrefab.prefab");
        Object.DestroyImmediate(swTemp);

        // Rocket platform prefab (variant of Platform with horizontal arrow + RocketPlatform comp)
        var rocketPlatformPrefab = BuildRocketPlatformPrefab(squareSprite);

        // Power-up Prefabs
        var springPrefab = BuildSpringPrefab(squareSprite);
        var shieldPrefab = BuildShieldPrefab(squareSprite, circleSprite);
        var jetpackPrefab = BuildJetpackPrefab(squareSprite);
        var multiplierCoinPrefab = BuildMultiplierCoinPrefab(squareSprite, circleSprite);

        // FX Prefabs
        Directory.CreateDirectory("Assets/Prefabs/FX");
        var bouncePrefab = BuildFxPrefab("BounceFX", burst: 10, speed: 5f, lifetime: 0.4f, size: 0.18f, gravity: 1.5f, radial: false);
        var switchPrefab = BuildFxPrefab("SwitchFX", burst: 24, speed: 6f, lifetime: 0.55f, size: 0.18f, gravity: 0.0f, radial: true);
        var deathPrefab  = BuildFxPrefab("DeathFX",  burst: 36, speed: 8f, lifetime: 0.7f,  size: 0.22f, gravity: 0.5f, radial: true);

        // GameManager
        var existingGM = GameObject.Find("GameManager");
        if (existingGM != null) Object.DestroyImmediate(existingGM);
        var gm = new GameObject("GameManager");
        var gmComp = gm.AddComponent<GameManager>();
        gmComp.player = player.transform;
        gmComp.cam = cam;

        // EffectsManager
        var existingFx = GameObject.Find("EffectsManager");
        if (existingFx != null) Object.DestroyImmediate(existingFx);
        var fxGO = new GameObject("EffectsManager");
        var fx = fxGO.AddComponent<EffectsManager>();
        fx.bouncePrefab = bouncePrefab;
        fx.switchPrefab = switchPrefab;
        fx.deathPrefab = deathPrefab;
        fx.cameraFollow = cf;

        // Spawner
        var existingSp = GameObject.Find("Spawner");
        if (existingSp != null) Object.DestroyImmediate(existingSp);
        var spawnerGO = new GameObject("Spawner");
        var sp = spawnerGO.AddComponent<PlatformSpawner>();
        sp.platformPrefab = platformPrefab;
        sp.rocketPlatformPrefab = rocketPlatformPrefab;
        sp.switcherPrefab = switcherPrefab;
        sp.springPrefab = springPrefab;
        sp.shieldPrefab = shieldPrefab;
        sp.jetpackPrefab = jetpackPrefab;
        sp.multiplierCoinPrefab = multiplierCoinPrefab;
        sp.cam = cam;
        sp.player = player.transform;

        cf.target = player.transform;

        // Starter platforms (all matching color so the run begins clean)
        ColorId startColor = pcomp.currentColor;
        float[] ys = new float[] { -3.3f, -1.6f, 0.2f, 1.9f, 3.5f, 5.2f };
        float[] xs = new float[] { 0f, -2.2f, 1.8f, -1.0f, 2.3f, -2.0f };
        for (int i = 0; i < ys.Length; i++)
        {
            var inst = (GameObject)PrefabUtility.InstantiatePrefab(platformPrefab);
            inst.transform.position = new Vector3(xs[i], ys[i], 0f);
            var pl = inst.GetComponent<SmallGame.Platform>();
            // First two match for safety; the rest mostly match.
            ColorId c = (i < 2 || Random.value < 0.7f) ? startColor : Palette.RandomOther(startColor);
            pl.SetColor(c);
        }

        // Canvas / UI
        var existingCanvas = GameObject.Find("Canvas");
        if (existingCanvas != null) Object.DestroyImmediate(existingCanvas);
        var canvasGO = new GameObject("Canvas", typeof(Canvas), typeof(CanvasScaler), typeof(GraphicRaycaster));
        var canvas = canvasGO.GetComponent<Canvas>();
        canvas.renderMode = RenderMode.ScreenSpaceOverlay;
        var scaler = canvasGO.GetComponent<CanvasScaler>();
        scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
        scaler.referenceResolution = new Vector2(1080f, 1920f);

        var existingES = Object.FindAnyObjectByType<EventSystem>();
        if (existingES != null) Object.DestroyImmediate(existingES.gameObject);
#if ENABLE_INPUT_SYSTEM
        new GameObject("EventSystem", typeof(EventSystem), typeof(InputSystemUIInputModule));
#else
        new GameObject("EventSystem", typeof(EventSystem), typeof(StandaloneInputModule));
#endif

        var scoreGO = MakeText("ScoreText", canvasGO.transform, "Score: 0", 48, TextAnchor.UpperLeft);
        var scoreRT = scoreGO.GetComponent<RectTransform>();
        scoreRT.anchorMin = new Vector2(0f, 1f); scoreRT.anchorMax = new Vector2(0f, 1f);
        scoreRT.pivot = new Vector2(0f, 1f);
        scoreRT.anchoredPosition = new Vector2(40f, -40f);
        scoreRT.sizeDelta = new Vector2(600f, 80f);

        var multGO = MakeText("MultiplierText", canvasGO.transform, "x2  0.0s", 44, TextAnchor.UpperCenter);
        var multRT = multGO.GetComponent<RectTransform>();
        multRT.anchorMin = new Vector2(0.5f, 1f); multRT.anchorMax = new Vector2(0.5f, 1f);
        multRT.pivot = new Vector2(0.5f, 1f);
        multRT.anchoredPosition = new Vector2(0f, -40f);
        multRT.sizeDelta = new Vector2(500f, 70f);
        multGO.GetComponent<Text>().color = new Color(1f, 0.85f, 0.2f);
        multGO.SetActive(false);

        var bestGO = MakeText("BestText", canvasGO.transform, "Best: 0", 40, TextAnchor.UpperRight);
        var bestRT = bestGO.GetComponent<RectTransform>();
        bestRT.anchorMin = new Vector2(1f, 1f); bestRT.anchorMax = new Vector2(1f, 1f);
        bestRT.pivot = new Vector2(1f, 1f);
        bestRT.anchoredPosition = new Vector2(-40f, -50f);
        bestRT.sizeDelta = new Vector2(500f, 70f);

        var panel = new GameObject("GameOverPanel", typeof(RectTransform), typeof(CanvasRenderer), typeof(Image));
        panel.transform.SetParent(canvasGO.transform, false);
        var prt = panel.GetComponent<RectTransform>();
        prt.anchorMin = Vector2.zero; prt.anchorMax = Vector2.one;
        prt.offsetMin = Vector2.zero; prt.offsetMax = Vector2.zero;
        panel.GetComponent<Image>().color = new Color(0f, 0f, 0f, 0.7f);

        var goTitle = MakeText("Title", panel.transform, "GAME OVER", 96, TextAnchor.MiddleCenter);
        SetCenter(goTitle, new Vector2(0f, 260f), new Vector2(900f, 140f));

        var newRecGO = MakeText("NewRecord", panel.transform, "NEW BEST!", 56, TextAnchor.MiddleCenter);
        SetCenter(newRecGO, new Vector2(0f, 150f), new Vector2(700f, 90f));
        newRecGO.GetComponent<Text>().color = new Color(0.97f, 0.85f, 0.25f);
        newRecGO.SetActive(false);

        var finalScoreGO = MakeText("FinalScore", panel.transform, "Score: 0   Best: 0", 52, TextAnchor.MiddleCenter);
        SetCenter(finalScoreGO, new Vector2(0f, 40f), new Vector2(900f, 100f));

        var btnGO = new GameObject("RestartButton", typeof(RectTransform), typeof(CanvasRenderer), typeof(Image), typeof(Button));
        btnGO.transform.SetParent(panel.transform, false);
        SetCenter(btnGO, new Vector2(0f, -140f), new Vector2(420f, 130f));
        btnGO.GetComponent<Image>().color = new Color(0.25f, 0.55f, 0.95f);
        var btnLabel = MakeText("Label", btnGO.transform, "Restart", 56, TextAnchor.MiddleCenter);
        var lrt = btnLabel.GetComponent<RectTransform>();
        lrt.anchorMin = Vector2.zero; lrt.anchorMax = Vector2.one;
        lrt.offsetMin = Vector2.zero; lrt.offsetMax = Vector2.zero;

        panel.SetActive(false);

        var ui = canvasGO.AddComponent<UIController>();
        ui.scoreText = scoreGO.GetComponent<Text>();
        ui.bestText = bestGO.GetComponent<Text>();
        ui.multiplierText = multGO.GetComponent<Text>();
        ui.gameOverPanel = panel;
        ui.finalScoreText = finalScoreGO.GetComponent<Text>();
        ui.newRecordLabel = newRecGO;
        ui.restartButton = btnGO.GetComponent<Button>();

        EditorSceneManager.MarkSceneDirty(SceneManager.GetActiveScene());
        EditorSceneManager.SaveScene(SceneManager.GetActiveScene());
        Debug.Log("[SmallGame] Scene built successfully.");
    }

    [MenuItem("SmallGame/Reset High Score")]
    public static void ResetHighScore()
    {
        PlayerPrefs.DeleteKey("SmallGame.HighScore");
        PlayerPrefs.Save();
        Debug.Log("[SmallGame] High score reset.");
    }

    static GameObject BuildRocketPlatformPrefab(Sprite squareSprite)
    {
        var go = new GameObject("RocketPlatformPrefab");
        go.transform.localScale = new Vector3(1.6f, 0.3f, 1f);
        var sr = go.AddComponent<SpriteRenderer>();
        sr.sprite = squareSprite;
        sr.sortingOrder = 5;
        var box = go.AddComponent<BoxCollider2D>();
        box.usedByEffector = true;
        var eff = go.AddComponent<PlatformEffector2D>();
        eff.useOneWay = true;
        eff.surfaceArc = 170f;
        go.AddComponent<SmallGame.Platform>();
        var rp = go.AddComponent<RocketPlatform>();

        // White arrow on top — 3 small squares forming a chevron pointing right
        var arrowParent = new GameObject("Arrow");
        arrowParent.transform.SetParent(go.transform, false);
        // Counter the platform's wide local scale so arrow stays roughly square
        arrowParent.transform.localScale = new Vector3(1f / 1.6f, 1f / 0.3f, 1f);
        arrowParent.transform.localPosition = new Vector3(0f, 0f, 0f);
        for (int i = 0; i < 3; i++)
        {
            var q = new GameObject("A" + i);
            q.transform.SetParent(arrowParent.transform, false);
            float xx = -0.18f + i * 0.18f;
            q.transform.localPosition = new Vector3(xx, 0f, 0f);
            q.transform.localScale = new Vector3(0.18f, 0.18f - i * 0.04f, 1f);
            var asr = q.AddComponent<SpriteRenderer>();
            asr.sprite = squareSprite;
            asr.color = Color.white;
            asr.sortingOrder = 7;
        }
        rp.arrowVisual = arrowParent.transform; // direction flips the whole arrow group

        var prefab = PrefabUtility.SaveAsPrefabAsset(go, "Assets/Prefabs/RocketPlatformPrefab.prefab");
        Object.DestroyImmediate(go);
        return prefab;
    }

    static GameObject BuildMultiplierCoinPrefab(Sprite squareSprite, Sprite circleSprite)
    {
        var go = new GameObject("MultiplierCoin");
        var col = go.AddComponent<CircleCollider2D>();
        col.isTrigger = true; col.radius = 0.4f;
        go.AddComponent<MultiplierCoin>();
        Color gold = new Color(1f, 0.85f, 0.2f);
        Color goldDark = new Color(0.75f, 0.55f, 0.1f);
        var outer = new GameObject("Outer");
        outer.transform.SetParent(go.transform, false);
        outer.transform.localScale = new Vector3(0.7f, 0.7f, 1f);
        var osr = outer.AddComponent<SpriteRenderer>();
        osr.sprite = circleSprite != null ? circleSprite : squareSprite;
        osr.color = gold;
        osr.sortingOrder = 6;
        var inner = new GameObject("Inner");
        inner.transform.SetParent(go.transform, false);
        inner.transform.localScale = new Vector3(0.5f, 0.5f, 1f);
        var isr = inner.AddComponent<SpriteRenderer>();
        isr.sprite = circleSprite != null ? circleSprite : squareSprite;
        isr.color = goldDark;
        isr.sortingOrder = 7;
        // Two small white pips to imply "x2"
        for (int i = 0; i < 2; i++)
        {
            var pip = new GameObject("Pip" + i);
            pip.transform.SetParent(go.transform, false);
            pip.transform.localPosition = new Vector3(-0.12f + i * 0.24f, 0f, 0f);
            pip.transform.localScale = new Vector3(0.12f, 0.18f, 1f);
            var psr = pip.AddComponent<SpriteRenderer>();
            psr.sprite = squareSprite;
            psr.color = Color.white;
            psr.sortingOrder = 8;
        }
        var prefab = PrefabUtility.SaveAsPrefabAsset(go, "Assets/Prefabs/MultiplierCoin.prefab");
        Object.DestroyImmediate(go);
        return prefab;
    }

    static GameObject BuildSpringPrefab(Sprite squareSprite)
    {
        var go = new GameObject("SpringPickup");
        var col = go.AddComponent<CircleCollider2D>();
        col.isTrigger = true; col.radius = 0.4f;
        go.AddComponent<SpringPickup>();
        // Green upward chevron from 3 squares
        Color green = new Color(0.35f, 0.95f, 0.45f);
        for (int i = 0; i < 3; i++)
        {
            var q = new GameObject("Bar" + i);
            q.transform.SetParent(go.transform, false);
            float yy = -0.15f + i * 0.18f;
            float w = 0.55f - i * 0.15f;
            q.transform.localPosition = new Vector3(0f, yy, 0f);
            q.transform.localScale = new Vector3(w, 0.12f, 1f);
            var sr = q.AddComponent<SpriteRenderer>();
            sr.sprite = squareSprite;
            sr.color = green;
            sr.sortingOrder = 6;
        }
        var prefab = PrefabUtility.SaveAsPrefabAsset(go, "Assets/Prefabs/SpringPickup.prefab");
        Object.DestroyImmediate(go);
        return prefab;
    }

    static GameObject BuildShieldPrefab(Sprite squareSprite, Sprite circleSprite)
    {
        var go = new GameObject("ShieldPickup");
        var col = go.AddComponent<CircleCollider2D>();
        col.isTrigger = true; col.radius = 0.4f;
        go.AddComponent<ShieldPickup>();
        Color cyan = new Color(0.55f, 0.95f, 1f);
        // Outer ring (cyan circle)
        var ring = new GameObject("Ring");
        ring.transform.SetParent(go.transform, false);
        ring.transform.localScale = new Vector3(0.85f, 0.85f, 1f);
        var rsr = ring.AddComponent<SpriteRenderer>();
        rsr.sprite = circleSprite != null ? circleSprite : squareSprite;
        rsr.color = cyan;
        rsr.sortingOrder = 6;
        // Inner darker hole
        var hole = new GameObject("Hole");
        hole.transform.SetParent(go.transform, false);
        hole.transform.localScale = new Vector3(0.55f, 0.55f, 1f);
        var hsr = hole.AddComponent<SpriteRenderer>();
        hsr.sprite = circleSprite != null ? circleSprite : squareSprite;
        hsr.color = new Color(0.12f, 0.12f, 0.16f);
        hsr.sortingOrder = 7;
        var prefab = PrefabUtility.SaveAsPrefabAsset(go, "Assets/Prefabs/ShieldPickup.prefab");
        Object.DestroyImmediate(go);
        return prefab;
    }

    static GameObject BuildJetpackPrefab(Sprite squareSprite)
    {
        var go = new GameObject("JetpackPickup");
        var col = go.AddComponent<CircleCollider2D>();
        col.isTrigger = true; col.radius = 0.4f;
        go.AddComponent<JetpackPickup>();
        Color orange = new Color(1f, 0.6f, 0.2f);
        // Body
        var body = new GameObject("Body");
        body.transform.SetParent(go.transform, false);
        body.transform.localScale = new Vector3(0.35f, 0.6f, 1f);
        var bsr = body.AddComponent<SpriteRenderer>();
        bsr.sprite = squareSprite;
        bsr.color = orange;
        bsr.sortingOrder = 6;
        // Flame
        var flame = new GameObject("Flame");
        flame.transform.SetParent(go.transform, false);
        flame.transform.localPosition = new Vector3(0f, -0.45f, 0f);
        flame.transform.localScale = new Vector3(0.22f, 0.28f, 1f);
        var fsr = flame.AddComponent<SpriteRenderer>();
        fsr.sprite = squareSprite;
        fsr.color = new Color(1f, 0.85f, 0.25f);
        fsr.sortingOrder = 7;
        var prefab = PrefabUtility.SaveAsPrefabAsset(go, "Assets/Prefabs/JetpackPickup.prefab");
        Object.DestroyImmediate(go);
        return prefab;
    }

    static GameObject BuildFxPrefab(string name, int burst, float speed, float lifetime, float size, float gravity, bool radial)
    {
        var go = new GameObject(name);
        var ps = go.AddComponent<ParticleSystem>();
        var em = ps.emission;
        em.enabled = true;
        em.rateOverTime = 0f;
        em.SetBursts(new ParticleSystem.Burst[] { new ParticleSystem.Burst(0f, (short)burst) });

        var main = ps.main;
        main.startLifetime = lifetime;
        main.startSpeed = speed;
        main.startSize = size;
        main.gravityModifier = gravity;
        main.startColor = Color.white;
        main.simulationSpace = ParticleSystemSimulationSpace.World;
        main.maxParticles = 200;
        main.duration = 0.2f;
        main.loop = false;

        var shape = ps.shape;
        shape.enabled = true;
        if (radial)
        {
            shape.shapeType = ParticleSystemShapeType.Circle;
            shape.radius = 0.05f;
            shape.arc = 360f;
        }
        else
        {
            shape.shapeType = ParticleSystemShapeType.Cone;
            shape.angle = 30f;
            shape.radius = 0.05f;
            shape.rotation = new Vector3(-90f, 0f, 0f); // emit upward
        }

        var col = ps.colorOverLifetime;
        col.enabled = true;
        var grad = new Gradient();
        grad.SetKeys(
            new GradientColorKey[] { new GradientColorKey(Color.white, 0f), new GradientColorKey(Color.white, 1f) },
            new GradientAlphaKey[] { new GradientAlphaKey(1f, 0f), new GradientAlphaKey(0f, 1f) }
        );
        col.color = new ParticleSystem.MinMaxGradient(grad);

        var sl = ps.sizeOverLifetime;
        sl.enabled = true;
        var sizeCurve = new AnimationCurve();
        sizeCurve.AddKey(0f, 1f);
        sizeCurve.AddKey(1f, 0f);
        sl.size = new ParticleSystem.MinMaxCurve(1f, sizeCurve);

        var psr = go.GetComponent<ParticleSystemRenderer>();
        psr.renderMode = ParticleSystemRenderMode.Billboard;
        psr.sortingOrder = 20;
        // Default-Particle material from built-in resources
        psr.material = AssetDatabase.GetBuiltinExtraResource<Material>("Default-Particle.mat");

        go.AddComponent<OneShotParticles>();

        var prefab = PrefabUtility.SaveAsPrefabAsset(go, "Assets/Prefabs/FX/" + name + ".prefab");
        Object.DestroyImmediate(go);
        return prefab;
    }

    static void SetCenter(GameObject go, Vector2 pos, Vector2 size)
    {
        var rt = go.GetComponent<RectTransform>();
        rt.anchorMin = new Vector2(0.5f, 0.5f);
        rt.anchorMax = new Vector2(0.5f, 0.5f);
        rt.pivot = new Vector2(0.5f, 0.5f);
        rt.anchoredPosition = pos;
        rt.sizeDelta = size;
    }

    static GameObject MakeText(string name, Transform parent, string content, int size, TextAnchor anchor)
    {
        var go = new GameObject(name, typeof(RectTransform), typeof(CanvasRenderer), typeof(Text));
        go.transform.SetParent(parent, false);
        var t = go.GetComponent<Text>();
        t.text = content;
        t.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        t.fontSize = size;
        t.alignment = anchor;
        t.color = Color.white;
        return go;
    }

    static Sprite GenerateCircleSprite()
    {
        string p = "Assets/Generated/Circle.png";
        Directory.CreateDirectory("Assets/Generated");
        if (!File.Exists(p))
        {
            int sz = 128;
            var tex = new Texture2D(sz, sz, TextureFormat.RGBA32, false);
            var px = new Color[sz * sz];
            float r = sz * 0.5f - 1f;
            for (int y = 0; y < sz; y++)
                for (int x = 0; x < sz; x++)
                {
                    float dx = x - sz * 0.5f + 0.5f, dy = y - sz * 0.5f + 0.5f;
                    px[y * sz + x] = (dx * dx + dy * dy <= r * r) ? Color.white : new Color(0f, 0f, 0f, 0f);
                }
            tex.SetPixels(px); tex.Apply();
            File.WriteAllBytes(p, tex.EncodeToPNG());
            AssetDatabase.ImportAsset(p);
            var ti = (TextureImporter)AssetImporter.GetAtPath(p);
            ti.textureType = TextureImporterType.Sprite;
            ti.spritePixelsPerUnit = 100;
            ti.SaveAndReimport();
        }
        return AssetDatabase.LoadAssetAtPath<Sprite>(p);
    }
}
#endif
