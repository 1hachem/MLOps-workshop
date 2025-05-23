# ML deployement with huggingface, fastapi, docker and ngrok

## requirements

you can install devbox to avoid having to install these tools manually

- uv
- go-task
- docker
- docker compose
- ngrok

## OPTIONAL installing devbox

- You can use Devbox on a Windows machine using Windows Subsystem for Linux 2.

Installing WSL2

To install WSL2 with the default Ubuntu distribution, open Powershell or Windows Command Prompt as an administrator, and run:

```
wsl --install
```

If WSL2 is already installed, you can install Ubuntu by running

```
wsl --install -d Ubuntu
```

If you are running an older version of Windows, you may need to follow the manual installation steps to enable virtualization and WSL2 on your system. See the official docs for more details

Run the following script in your WSL2 terminal as a non-root user to install Devbox.

```
curl -fsSL https://get.jetify.com/devbox | bash
```

Devbox requires the Nix Package Manager. If Nix is not detected on your machine when running a command, Devbox will automatically install it in single user mode for WSL2. Don't worry: You can use Devbox without needing to learn the Nix Language.
