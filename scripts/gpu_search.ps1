# Exhaustive GPU search across ALL cloud types with popular image
$apiKey = "$env:RUNPOD_API_KEY"
$result = @()

$gpuIds = @(
    "NVIDIA RTX A6000",
    "NVIDIA A40",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA GeForce RTX 3090 Ti",
    "NVIDIA RTX A5000",
    "NVIDIA L40S",
    "NVIDIA L40",
    "NVIDIA L4",
    "NVIDIA RTX 6000 Ada Generation",
    "NVIDIA RTX 5000 Ada Generation",
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA A100 80GB PCIe",
    "NVIDIA H100 80GB HBM3",
    "NVIDIA RTX 4000 Ada Generation"
)

foreach ($gpu in $gpuIds) {
    foreach ($cloud in @("COMMUNITY","SECURE")) {
        $b = '{"query":"mutation{podFindAndDeployOnDemand(input:{name:\"pampar\",gpuTypeId:\"'+$gpu+'\",gpuCount:1,cloudType:'+$cloud+',volumeInGb:30,containerDiskInGb:10,imageName:\"runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04\",startSsh:true,env:[{key:\"PUBLIC_KEY\",value:\"ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFJGhFmO0MFzNGVwvJrfG+sAoKfalj5rcOOVdiqxBMwR lucas@DESKTOP-LUCAS\"}]}){id desiredStatus}}"}'
        try {
            $r = Invoke-RestMethod -Uri "https://api.runpod.io/graphql?api_key=$apiKey" -Method POST -Headers @{"Content-Type"="application/json"} -Body $b -TimeoutSec 10
            if (-not $r.errors) {
                $id = $r.data.podFindAndDeployOnDemand.id
                $result += "SUCCESS|$gpu|$cloud|$id"
                # Immediately stop - we got one
                $result | Out-File C:\Users\lucas\Documents\gpu_search.txt
                exit 0
            } else {
                $result += "FAIL|$gpu|$cloud|$($r.errors[0].message)"
            }
        } catch {
            $result += "ERROR|$gpu|$cloud|$($_.Exception.Message)"
        }
    }
}

$result | Out-File C:\Users\lucas\Documents\gpu_search.txt
