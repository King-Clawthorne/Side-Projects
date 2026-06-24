using UnrealBuildTool;

public class BlazePoseLiveLink : ModuleRules
{
    public BlazePoseLiveLink(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;

        PublicDependencyModuleNames.AddRange(new string[]
        {
            "Core",
            "CoreUObject",
            "Engine",
            "LiveLink",
            "LiveLinkInterface",
            "Networking",
            "Sockets",
        });

        PrivateDependencyModuleNames.AddRange(new string[]
        {
        });
    }
}
