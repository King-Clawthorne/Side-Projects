#include "BlazePoseLiveLinkSourceFactory.h"
#include "BlazePoseLiveLinkSource.h"
#include "Interfaces/IPv4/IPv4Address.h"
#include "Interfaces/IPv4/IPv4Endpoint.h"

#define LOCTEXT_NAMESPACE "BlazePoseLiveLinkSourceFactory"

// Default UDP port matches python/pose_sender.py default. Edit and rebuild
// to change, or pass "Port=NNNN" via ConnectionString from a custom panel.
static constexpr uint16 DefaultPort = 14043;

FText UBlazePoseLiveLinkSourceFactory::GetSourceDisplayName() const
{
    return LOCTEXT("DisplayName", "BlazePose UDP");
}

FText UBlazePoseLiveLinkSourceFactory::GetSourceTooltip() const
{
    return LOCTEXT("Tooltip",
        "Listens for BlazePose pose data over UDP (default port 14043) and "
        "publishes it as a Live Link animation subject.");
}

ULiveLinkSourceFactory::EMenuType UBlazePoseLiveLinkSourceFactory::GetMenuType() const
{
    return EMenuType::MenuEntry;
}

TSharedPtr<ILiveLinkSource>
UBlazePoseLiveLinkSourceFactory::CreateSource(const FString& ConnectionString) const
{
    int32 Port = DefaultPort;
    if (!ConnectionString.IsEmpty())
    {
        FParse::Value(*ConnectionString, TEXT("Port="), Port);
    }

    const FIPv4Endpoint Endpoint(FIPv4Address::Any, static_cast<uint16>(Port));
    return MakeShared<FBlazePoseLiveLinkSource>(Endpoint);
}

#undef LOCTEXT_NAMESPACE
