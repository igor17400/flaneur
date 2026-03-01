output "public_ip" {
  description = "Public IP of the Derive server"
  value       = aws_eip.derive.public_ip
}

output "url" {
  description = "URL to access Derive"
  value       = "http://${aws_eip.derive.public_ip}"
}

output "ssh" {
  description = "SSH command to connect"
  value       = "ssh -i <your-key>.pem ubuntu@${aws_eip.derive.public_ip}"
}

output "setup_log" {
  description = "Command to check setup progress"
  value       = "ssh ubuntu@${aws_eip.derive.public_ip} 'sudo tail -f /var/log/derive-setup.log'"
}
