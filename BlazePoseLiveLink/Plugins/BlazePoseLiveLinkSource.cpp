#include "BlazePoseLiveLinkSource.h"

#include "ILiveLinkClient.h"
#include "Roles/LiveLinkTransformRole.h"
#include "Roles/LiveLinkTransformTypes.h"

#include "Common/UdpSocketBuilder.h"
#include "Sockets.h"
#include "SocketSubsystem.h"
#include "HAL/RunnableThread.h"
#include "HAL/PlatformProcess.h"
#include "Interfaces/IPv4/IPv4Address.h"

#define LOCTEXT_NAMESPACE "BlazePoseLiveLinkSource"

namespace BlazePoseProtocol
{
    static const uint8 PROTO_MAGIC[4] = { 'U', 'E', 'L', 'P' };
    constexpr uint8  PROTO_VERSION = 1;
    constexpr int32  JOINT_COUNT   = 34;
    constexpr int32  HEADER_SIZE   = 19; // 4 + 1 + 4 + 8 + 2

    // Keep these in sync with python/pose_sender.py and test_receiver.py.
    static const TArray<FName>& BoneNames()
    {
        static const TArray<FName> Names = {
            TEXT("pelvis"),
            TEXT("nose"),
            TEXT("l_eye_inner"), TEXT("l_eye"), TEXT("l_eye_outer"),
            TEXT("r_eye_inner"), TEXT("r_eye"), TEXT("r_eye_outer"),
            TEXT("l_ear"), TEXT("r_ear"),
            TEXT("mouth_l"), TEXT("mouth_r"),
            TEXT("l_shoulder"), TEXT("r_shoulder"),
            TEXT("l_elbow"), TEXT("r_elbow"),
            TEXT("l_wrist"), TEXT("r_wrist"),
            TEXT("l_pinky"), TEXT("r_pinky"),
            TEXT("l_index"), TEXT("r_index"),
            TEXT("l_thumb"), TEXT("r_thumb"),
            TEXT("l_hip"),   TEXT("r_hip"),
            TEXT("l_knee"),  TEXT("r_knee"),
            TEXT("l_ankle"), TEXT("r_ankle"),
            TEXT("l_heel"),  TEXT("r_heel"),
            TEXT("l_foot_idx"), TEXT("r_foot_idx"),
        };
        return Names;
    }

    static const FString& SubjectPrefix()
    {
        static const FString Prefix(TEXT("BlazePose_"));
        return Prefix;
    }
}

FBlazePoseLiveLinkSource::FBlazePoseLiveLinkSource(const FIPv4Endpoint& InEndpoint)
    : Endpoint(InEndpoint)
    , bStopping(false)
{
    using namespace BlazePoseProtocol;
    const TArray<FName>& Bones = BoneNames();
    MarkerSubjectNames.Reserve(Bones.Num());
    for (const FName& Bone : Bones)
    {
        MarkerSubjectNames.Add(FName(*(SubjectPrefix() + Bone.ToString())));
    }
}

FBlazePoseLiveLinkSource::~FBlazePoseLiveLinkSource()
{
    Cleanup();
}

void FBlazePoseLiveLinkSource::Cleanup()
{
    bStopping = true;

    if (Thread)
    {
        Thread->Kill(true);
        delete Thread;
        Thread = nullptr;
    }

    if (Socket)
    {
        Socket->Close();
        ISocketSubsystem* SocketSubsystem = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM);
        if (SocketSubsystem)
        {
            SocketSubsystem->DestroySocket(Socket);
        }
        Socket = nullptr;
    }
}

void FBlazePoseLiveLinkSource::ReceiveClient(ILiveLinkClient* InClient, FGuid InSourceGuid)
{
    Client = InClient;
    SourceGuid = InSourceGuid;

    Socket = FUdpSocketBuilder(TEXT("BlazePoseLiveLinkSocket"))
        .AsReusable()
        .BoundToEndpoint(Endpoint)
        .WithReceiveBufferSize(2 * 1024 * 1024)
        .Build();

    if (!Socket)
    {
        UE_LOG(LogTemp, Error,
            TEXT("BlazePoseLiveLink: failed to bind UDP %s"),
            *Endpoint.ToString());
        return;
    }

    Thread = FRunnableThread::Create(
        this, TEXT("BlazePoseLiveLinkRecv"), 0, TPri_AboveNormal);
}

bool FBlazePoseLiveLinkSource::IsSourceStillValid() const
{
    return Socket != nullptr && !bStopping;
}

bool FBlazePoseLiveLinkSource::RequestSourceShutdown()
{
    Cleanup();
    return true;
}

FText FBlazePoseLiveLinkSource::GetSourceType() const
{
    return LOCTEXT("SourceType", "BlazePose UDP");
}

FText FBlazePoseLiveLinkSource::GetSourceMachineName() const
{
    return FText::FromString(Endpoint.ToString());
}

FText FBlazePoseLiveLinkSource::GetSourceStatus() const
{
    return FText::FromString(FString::Printf(
        TEXT("Frames: %d"), FramesReceivedCounter.GetValue()));
}

bool FBlazePoseLiveLinkSource::Init() { return true; }
void FBlazePoseLiveLinkSource::Stop() { bStopping = true; }
void FBlazePoseLiveLinkSource::Exit() {}

uint32 FBlazePoseLiveLinkSource::Run()
{
    using namespace BlazePoseProtocol;

    TArray<uint8> Buffer;
    Buffer.SetNumUninitialized(8192);

    ISocketSubsystem* SocketSubsystem = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM);
    TSharedRef<FInternetAddr> Sender = SocketSubsystem->CreateInternetAddr();

    while (!bStopping)
    {
        if (!Socket) { break; }

        uint32 Pending = 0;
        if (Socket->HasPendingData(Pending))
        {
            int32 BytesRead = 0;
            if (Socket->RecvFrom(Buffer.GetData(), Buffer.Num(), BytesRead, *Sender)
                && BytesRead > 0)
            {
                HandleDatagram(Buffer.GetData(), BytesRead);
            }
        }
        else
        {
            // Short sleep to avoid busy-spinning while still keeping latency low.
            FPlatformProcess::Sleep(0.0005f);
        }
    }

    return 0;
}

void FBlazePoseLiveLinkSource::PushStaticTransforms()
{
    using namespace BlazePoseProtocol;

    if (!Client) { return; }

    for (const FName& SubjectName : MarkerSubjectNames)
    {
        FLiveLinkStaticDataStruct StaticData(FLiveLinkTransformStaticData::StaticStruct());
        const FLiveLinkSubjectKey Key(SourceGuid, SubjectName);
        Client->RemoveSubject_AnyThread(Key);
        Client->PushSubjectStaticData_AnyThread(
            Key, ULiveLinkTransformRole::StaticClass(), MoveTemp(StaticData));
    }

    bStaticPushed = true;
}

void FBlazePoseLiveLinkSource::HandleDatagram(const uint8* Data, int32 NumBytes)
{
    using namespace BlazePoseProtocol;

    if (NumBytes < HEADER_SIZE) { return; }

    const uint8* P = Data;

    // Magic
    if (FMemory::Memcmp(P, PROTO_MAGIC, 4) != 0) { return; }
    P += 4;

    // Version
    const uint8 Version = *P++;
    if (Version != PROTO_VERSION) { return; }

    // FrameId, Timestamp, JointCount (little-endian, packed)
    uint32 FrameId   = 0; FMemory::Memcpy(&FrameId,   P, 4); P += 4;
    double Timestamp = 0; FMemory::Memcpy(&Timestamp, P, 8); P += 8;
    uint16 JointCount = 0; FMemory::Memcpy(&JointCount, P, 2); P += 2;

    if (JointCount != JOINT_COUNT) { return; }

    const int32 BodyBytes = JOINT_COUNT * 4 * sizeof(float);
    if (NumBytes - HEADER_SIZE < BodyBytes) { return; }

    if (!Client) { return; }

    if (!bStaticPushed)
    {
        PushStaticTransforms();
    }

    // Parse joints (x, y, z, visibility) per joint, all f32 little-endian.
    // Each joint is published as its own Transform subject in world space,
    // so it can drive a marker Actor via a Live Link Transform Controller.
    const float* F = reinterpret_cast<const float*>(P);
    const double Now = FPlatformTime::Seconds();

    for (int32 i = 0; i < JOINT_COUNT; ++i)
    {
        const FVector World(F[i * 4 + 0], F[i * 4 + 1], F[i * 4 + 2]);

        FLiveLinkFrameDataStruct FrameData(FLiveLinkTransformFrameData::StaticStruct());
        FLiveLinkTransformFrameData& Xf = *FrameData.Cast<FLiveLinkTransformFrameData>();
        Xf.Transform = FTransform(FQuat::Identity, World, FVector::OneVector);
        Xf.WorldTime = Now;
        Xf.MetaData.SceneTime = FQualifiedFrameTime();

        const FLiveLinkSubjectKey Key(SourceGuid, MarkerSubjectNames[i]);
        Client->PushSubjectFrameData_AnyThread(Key, MoveTemp(FrameData));
    }

    FramesReceivedCounter.Increment();
}

#undef LOCTEXT_NAMESPACE
