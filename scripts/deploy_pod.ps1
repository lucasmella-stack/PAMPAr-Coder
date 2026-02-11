# Terminate stuck pod, try SECURE cloud with multiple GPU types
$apiKey = "$env:RUNPOD_API_KEY"

# Kill stuck pod
$bt = '{"query":"mutation{podTerminate(input:{podId:\"e3uosu5ubpkn2q\"})}"}'
Invoke-RestMethod -Uri "https://api.runpod.io/graphql?api_key=$apiKey" -Method POST -Headers @{"Content-Type"="application/json"} -Body $bt | Out-Null
"Terminated e3uosu5ubpkn2q" | Out-File C:\Users\lucas\Documents\pod_result.txt

# Try SECURE cloud GPUs
$gpus = @("NVIDIA RTX A6000","NVIDIA A40","NVIDIA RTX 4090","NVIDIA GeForce RTX 3090","NVIDIA L40S","NVIDIA RTX A5000")
foreach ($gpu in $gpus) {
    $b = '{"query":"mutation{podFindAndDeployOnDemand(input:{name:\"pampar-secure\",gpuTypeId:\"'+$gpu+'\",gpuCount:1,cloudType:SECURE,volumeInGb:50,containerDiskInGb:20,imageName:\"runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04\",startSsh:true,env:[{key:\"PUBLIC_KEY\",value:\"ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFJGhFmO0MFzNGVwvJrfG+sAoKfalj5rcOOVdiqxBMwR lucas@DESKTOP-LUCAS\"}]}){id desiredStatus}}"}'
    $r = Invoke-RestMethod -Uri "https://api.runpod.io/graphql?api_key=$apiKey" -Method POST -Headers @{"Content-Type"="application/json"} -Body $b
    if (-not $r.errors) {
        "SUCCESS: $gpu => $($r.data.podFindAndDeployOnDemand.id)" | Add-Content C:\Users\lucas\Documents\pod_result.txt
        break
    } else {
        "FAIL: $gpu - $($r.errors[0].message)" | Add-Content C:\Users\lucas\Documents\pod_result.txt
    }
}
