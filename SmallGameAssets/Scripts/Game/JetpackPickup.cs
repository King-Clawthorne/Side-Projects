using UnityEngine;

namespace SmallGame
{
    public class JetpackPickup : PowerupPickup
    {
        public float duration = 1.2f;
        public float upwardSpeed = 22f;
        public override PowerupKind Kind => PowerupKind.Jetpack;
        public override Color Tint => new Color(1f, 0.6f, 0.2f);
        protected override void Apply(PlayerController player) => player.ActivateJetpack(duration, upwardSpeed);
    }
}
