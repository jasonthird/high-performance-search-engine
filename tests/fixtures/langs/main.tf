resource "aws_instance" "web" {
  ami           = "ami-123"
  instance_type = "t3.micro"
}

variable "width" {
  type    = number
  default = 3
}
