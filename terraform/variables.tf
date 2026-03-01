variable "aws_region" {
  description = "AWS region to deploy in"
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = "EC2 instance type (t3.small = 2GB RAM, enough for data loading)"
  type        = string
  default     = "t3.small"
}

variable "key_name" {
  description = "Name of an existing EC2 key pair for SSH access"
  type        = string
}

variable "allowed_ssh_cidr" {
  description = "CIDR block allowed to SSH (e.g. your IP: 1.2.3.4/32)"
  type        = string
  default     = "0.0.0.0/0"
}

variable "mistral_api_key" {
  description = "Mistral API key for the explain endpoint (optional)"
  type        = string
  default     = ""
  sensitive   = true
}

variable "repo_url" {
  description = "Git repo URL to clone"
  type        = string
  default     = "https://github.com/igor17400/flaneur.git"
}

variable "repo_branch" {
  description = "Git branch to deploy"
  type        = string
  default     = "main"
}
