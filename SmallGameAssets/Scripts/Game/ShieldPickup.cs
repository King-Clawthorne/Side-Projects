using UnityEngine;

namespace SmallGame
{
    public class ShieldPickup : PowerupPickup
    {
        public override PowerupKind Kind => PowerupKind.Shield;
        public override Color Tint => new Color(0.55f, 0.95f, 1f);
        protected override void Apply(PlayerController player) => player.GrantShield();
    }
}
