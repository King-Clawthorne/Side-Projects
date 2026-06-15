using UnityEngine;

namespace SmallGame
{
    [RequireComponent(typeof(ParticleSystem))]
    public class OneShotParticles : MonoBehaviour
    {
        ParticleSystem ps;
        void Awake() { ps = GetComponent<ParticleSystem>(); }
        void Update()
        {
            if (ps == null || (!ps.IsAlive(true)))
                Destroy(gameObject);
        }
    }
}
