#pragma once

#include "CoreMinimal.h"
#include "ILiveLinkSource.h"
#include "HAL/Runnable.h"
#include "HAL/ThreadSafeBool.h"
#include "Interfaces/IPv4/IPv4Endpoint.h"

class FSocket;
class FRunnableThread;
class ILiveLinkClient;

/**
 * Live Link source that listens for BlazePose pose frames over UDP and
 * publishes one Transform subject per joint ("BlazePose_pelvis",
 * "BlazePose_nose", ...). Each subject can drive a marker Actor (sphere,
 * cube, etc.) in the level via a Live Link Transform Controller — no
 * skeletal mesh, IK Rig, or retargeter required. Intended for
 * visualization of the raw landmark cloud.
 */
class FBlazePoseLiveLinkSource
    : public ILiveLinkSource
    , public FRunnable
{
public:
    explicit FBlazePoseLiveLinkSource(const FIPv4Endpoint& InEndpoint);
    virtual ~FBlazePoseLiveLinkSource();

    // ILiveLinkSource
    virtual void ReceiveClient(ILiveLinkClient* InClient, FGuid InSourceGuid) override;
    virtual bool IsSourceStillValid() const override;
    virtual bool RequestSourceShutdown() override;
    virtual FText GetSourceType() const override;
    virtual FText GetSourceMachineName() const override;
    virtual FText GetSourceStatus() const override;

    // FRunnable
    virtual bool Init() override;
    virtual uint32 Run() override;
    virtual void Stop() override;
    virtual void Exit() override;

private:
    void HandleDatagram(const uint8* Data, int32 NumBytes);
    void PushStaticTransforms();
    void Cleanup();

    ILiveLinkClient* Client = nullptr;
    FGuid SourceGuid;
    TArray<FName> MarkerSubjectNames;

    FIPv4Endpoint Endpoint;
    FSocket* Socket = nullptr;
    FRunnableThread* Thread = nullptr;
    FThreadSafeBool bStopping;
    bool bStaticPushed = false;

    FThreadSafeCounter FramesReceivedCounter;
};
