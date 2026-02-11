$body='{"query":"query{pod(input:{podId:\"adhlm9i4qd4uuv\"}){runtime{ports{ip privatePort publicPort}}}}"}'
$r=Invoke-RestMethod -Uri "https://api.runpod.io/graphql?api_key=$env:RUNPOD_API_KEY" -Method POST -Headers @{"Content-Type"="application/json"} -Body $body
$ssh=$r.data.pod.runtime.ports|Where-Object{$_.privatePort -eq 22}
if($ssh){
    "SSH_READY" | Out-File C:\Users\lucas\Documents\pod_ssh.txt
    "$($ssh.ip):$($ssh.publicPort)" | Add-Content C:\Users\lucas\Documents\pod_ssh.txt
} else {
    "NO_RUNTIME" | Out-File C:\Users\lucas\Documents\pod_ssh.txt
}
