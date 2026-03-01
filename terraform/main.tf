terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# --- Networking ---

resource "aws_security_group" "derive" {
  name        = "derive-sg"
  description = "Allow HTTP (80) and SSH (22)"

  ingress {
    description = "HTTP from anywhere"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "derive-sg" }
}

# --- EC2 Instance ---

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

resource "aws_instance" "derive" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.instance_type
  key_name               = var.key_name
  vpc_security_group_ids = [aws_security_group.derive.id]

  root_block_device {
    volume_size = 20
    volume_type = "gp3"
  }

  user_data = base64encode(templatefile("${path.module}/user_data.sh", {
    repo_url        = var.repo_url
    repo_branch     = var.repo_branch
    mistral_api_key = var.mistral_api_key
  }))

  tags = { Name = "derive-server" }
}

# --- Elastic IP (stable public address) ---

resource "aws_eip" "derive" {
  instance = aws_instance.derive.id
  tags     = { Name = "derive-eip" }
}
