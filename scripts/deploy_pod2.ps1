# Terminate stuck pod, try lighter image
$apiKey = "$env:RUNPOD_API_KEY"

# Kill stuck pod  
$bt = '{"query":"mutation{podTerminate(input:{podId:\"38namntmpqnop3\"})}"}'
Invoke-RestMethod -Uri "https://api.runpod.io/graphql?api_key=$apiKey" -Method POST -Headers @{"Content-Type"="application/json"} -Body $bt | Out-Null
"Terminated 38namntmpqnop3" | Out-File C:\Users\lucas\Documents\pod_result2.txt

# Try with lighter image (pytorch runtime, not devel) - faster pull
$gpus = @("NVIDIA RTX A6000","NVIDIA A40","NVIDIA GeForce RTX 4090","NVIDIA GeForce RTX 3090","NVIDIA L40S")
foreach ($gpu in $gpus) {
    $b = '{"query":"mutation{podFindAndDeployOnDemand(input:{name:\"pampar2\",gpuTypeId:\"'+$gpu+'\",gpuCount:1,cloudType:ALL,volumeInGb:50,containerDiskInGb:20,imageName:\"runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04\",startSsh:true,env:[{key:\"PUBLIC_KEY\",value:\"ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFJGhFmO0MFzNGVwvJrfG+sAoKfalj5rcOOVdiqxBMwR lucas@DESKTOP-LUCAS\"}]}){id desiredStatus}}"}'
    $r = Invoke-RestMethod -Uri "https://api.runpod.io/graphql?api_key=$apiKey" -Method POST -Headers @{"Content-Type"="application/json"} -Body $b
    if (-not $r.errors) {
        "SUCCESS: $gpu => $($r.data.podFindAndDeployOnDemand.id)" | Add-Content C:\Users\lucas\Documents\pod_result2.txt
        break
    } else {
        "FAIL: $gpu - $($r.errors[0].message)" | Add-Content C:\Users\lucas\Documents\pod_result2.txt
    }
}
