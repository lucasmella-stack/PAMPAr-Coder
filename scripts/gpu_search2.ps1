# Focused GPU search with 60s timeout
$apiKey = "$env:RUNPOD_API_KEY"

# First check if any ghost pods exist
$body='{"query":"query{myself{pods{id name desiredStatus}}}"}'
$r = Invoke-RestMethod -Uri "https://api.runpod.io/graphql?api_key=$apiKey" -Method POST -Headers @{"Content-Type"="application/json"} -Body $body -TimeoutSec 30
$existing = $r.data.myself.pods
if ($existing) {
    "EXISTING PODS:" | Out-File C:\Users\lucas\Documents\gpu_search2.txt
    foreach ($p in $existing) {
        "  $($p.id) | $($p.name) | $($p.desiredStatus)" | Add-Content C:\Users\lucas\Documents\gpu_search2.txt
        # Terminate ghost pods
        $bt = '{"query":"mutation{podTerminate(input:{podId:\"'+$p.id+'\"})}"}'
        Invoke-RestMethod -Uri "https://api.runpod.io/graphql?api_key=$apiKey" -Method POST -Headers @{"Content-Type"="application/json"} -Body $bt -TimeoutSec 15 | Out-Null
        "  -> Terminated" | Add-Content C:\Users\lucas\Documents\gpu_search2.txt
    }
} else {
    "No existing pods" | Out-File C:\Users\lucas\Documents\gpu_search2.txt
}

# Now try to create - fewer GPUs, 60s timeout
$gpus = @("NVIDIA GeForce RTX 4090","NVIDIA RTX A6000","NVIDIA A40","NVIDIA GeForce RTX 3090")
foreach ($gpu in $gpus) {
    "Trying $gpu..." | Add-Content C:\Users\lucas\Documents\gpu_search2.txt
    $b = '{"query":"mutation{podFindAndDeployOnDemand(input:{name:\"pampar\",gpuTypeId:\"'+$gpu+'\",gpuCount:1,cloudType:ALL,volumeInGb:30,containerDiskInGb:10,imageName:\"runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04\",startSsh:true,env:[{key:\"PUBLIC_KEY\",value:\"ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFJGhFmO0MFzNGVwvJrfG+sAoKfalj5rcOOVdiqxBMwR lucas@DESKTOP-LUCAS\"}]}){id desiredStatus}}"}'
    try {
        $r = Invoke-RestMethod -Uri "https://api.runpod.io/graphql?api_key=$apiKey" -Method POST -Headers @{"Content-Type"="application/json"} -Body $b -TimeoutSec 60
        if (-not $r.errors) {
            "SUCCESS: $gpu => $($r.data.podFindAndDeployOnDemand.id)" | Add-Content C:\Users\lucas\Documents\gpu_search2.txt
            exit 0
        } else {
            "FAIL: $gpu - $($r.errors[0].message)" | Add-Content C:\Users\lucas\Documents\gpu_search2.txt
        }
    } catch {
        "ERROR: $gpu - $($_.Exception.Message)" | Add-Content C:\Users\lucas\Documents\gpu_search2.txt
    }
}
"ALL FAILED" | Add-Content C:\Users\lucas\Documents\gpu_search2.txt
