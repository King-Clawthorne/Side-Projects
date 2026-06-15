using UnityEngine;

namespace SmallGame
{
    public class SpringPickup : PowerupPickup
    {
        public float bounceMultiplier = 2.1f;
        public override PowerupKind Kind => PowerupKind.Spring;
        public override Color Tint => new Color(0.35f, 0.95f, 0.45f);
        protected override void Apply(PlayerController player) => player.QueueBounceBoost(bounceMultiplier);
    }
}
