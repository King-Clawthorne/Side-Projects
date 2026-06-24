#pragma once

#include "CoreMinimal.h"
#include "LiveLinkSourceFactory.h"
#include "BlazePoseLiveLinkSourceFactory.generated.h"

UCLASS()
class UBlazePoseLiveLinkSourceFactory : public ULiveLinkSourceFactory
{
    GENERATED_BODY()

public:
    virtual FText GetSourceDisplayName() const override;
    virtual FText GetSourceTooltip() const override;
    virtual EMenuType GetMenuType() const override;
    virtual TSharedPtr<ILiveLinkSource> CreateSource(const FString& ConnectionString) const override;
};
