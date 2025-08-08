"""Automatic Ollama installation helper."""

import subprocess
import platform
import logging
import os
import time
import asyncio
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class OllamaInstaller:
    """Handle automatic Ollama installation."""
    
    @staticmethod
    def check_ollama_installed() -> bool:
        """Check if Ollama is installed."""
        try:
            result = subprocess.run(
                ["which", "ollama"],
                capture_output=True,
                text=True,
                check=False
            )
            return result.returncode == 0
        except Exception:
            return False
    
    @staticmethod
    def check_ollama_running() -> bool:
        """Check if Ollama service is running."""
        try:
            result = subprocess.run(
                ["curl", "-s", "http://localhost:11434/api/tags"],
                capture_output=True,
                text=True,
                check=False
            )
            return result.returncode == 0
        except Exception:
            return False
    
    @staticmethod
    async def install_ollama() -> bool:
        """Install Ollama based on the platform."""
        system = platform.system().lower()
        
        if system == "darwin":  # macOS
            return await OllamaInstaller._install_ollama_macos()
        elif system == "linux":
            return await OllamaInstaller._install_ollama_linux()
        else:
            logger.error(f"Unsupported platform for automatic Ollama installation: {system}")
            return False
    
    @staticmethod
    async def _install_ollama_macos() -> bool:
        """Install Ollama on macOS."""
        logger.info("Installing Ollama on macOS...")
        
        try:
            # Check if Homebrew is installed
            brew_check = subprocess.run(
                ["which", "brew"],
                capture_output=True,
                text=True,
                check=False
            )
            
            if brew_check.returncode == 0:
                # Install via Homebrew
                logger.info("Installing Ollama via Homebrew...")
                result = subprocess.run(
                    ["brew", "install", "ollama"],
                    capture_output=True,
                    text=True,
                    check=False
                )
                
                if result.returncode == 0:
                    logger.info("Ollama installed successfully via Homebrew")
                    return True
                else:
                    logger.warning("Homebrew installation failed, trying direct download...")
            
            # Direct download and install
            logger.info("Downloading Ollama installer...")
            download_cmd = [
                "curl", "-L", "-o", "/tmp/ollama-darwin",
                "https://ollama.com/download/ollama-darwin"
            ]
            
            result = subprocess.run(download_cmd, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                logger.error("Failed to download Ollama")
                return False
            
            # Make executable and move to /usr/local/bin
            logger.info("Installing Ollama to /usr/local/bin...")
            commands = [
                ["chmod", "+x", "/tmp/ollama-darwin"],
                ["sudo", "mv", "/tmp/ollama-darwin", "/usr/local/bin/ollama"]
            ]
            
            for cmd in commands:
                result = subprocess.run(cmd, capture_output=True, text=True, check=False)
                if result.returncode != 0:
                    logger.error(f"Failed to execute: {' '.join(cmd)}")
                    return False
            
            logger.info("Ollama installed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error installing Ollama on macOS: {e}")
            return False
    
    @staticmethod
    async def _install_ollama_linux() -> bool:
        """Install Ollama on Linux."""
        logger.info("Installing Ollama on Linux...")
        
        try:
            # Use the official install script
            install_cmd = "curl -fsSL https://ollama.com/install.sh | sh"
            
            result = subprocess.run(
                install_cmd,
                shell=True,
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                logger.info("Ollama installed successfully on Linux")
                return True
            else:
                logger.error(f"Failed to install Ollama: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error installing Ollama on Linux: {e}")
            return False
    
    @staticmethod
    async def start_ollama_service() -> bool:
        """Start the Ollama service."""
        logger.info("Starting Ollama service...")
        
        try:
            system = platform.system().lower()
            
            if system == "darwin":
                # On macOS, start Ollama in background
                subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True
                )
            elif system == "linux":
                # On Linux, use systemctl if available
                systemctl_check = subprocess.run(
                    ["which", "systemctl"],
                    capture_output=True,
                    text=True,
                    check=False
                )
                
                if systemctl_check.returncode == 0:
                    subprocess.run(
                        ["sudo", "systemctl", "start", "ollama"],
                        capture_output=True,
                        check=False
                    )
                else:
                    # Fallback to running directly
                    subprocess.Popen(
                        ["ollama", "serve"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True
                    )
            
            # Wait for service to start
            for _ in range(30):  # Wait up to 30 seconds
                await asyncio.sleep(1)
                if OllamaInstaller.check_ollama_running():
                    logger.info("Ollama service started successfully")
                    return True
            
            logger.error("Ollama service failed to start within timeout")
            return False
            
        except Exception as e:
            logger.error(f"Error starting Ollama service: {e}")
            return False
    
    @staticmethod
    async def pull_model(model_name: str) -> bool:
        """Pull a specific Ollama model."""
        logger.info(f"Pulling Ollama model: {model_name}")
        
        try:
            # Run ollama pull command
            result = await asyncio.create_subprocess_exec(
                "ollama", "pull", model_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0:
                logger.info(f"Successfully pulled model: {model_name}")
                return True
            else:
                logger.error(f"Failed to pull model: {stderr.decode()}")
                return False
                
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            return False
    
    @staticmethod
    async def ensure_ollama_ready(model_name: str) -> Tuple[bool, str]:
        """
        Ensure Ollama is installed, running, and has the required model.
        
        Returns:
            Tuple of (success, message)
        """
        # Check if Ollama is installed
        if not OllamaInstaller.check_ollama_installed():
            logger.info("Ollama not found, attempting automatic installation...")
            
            # Try to install Ollama
            if not await OllamaInstaller.install_ollama():
                return False, "Failed to install Ollama. Please install manually from https://ollama.com/download"
        
        # Check if Ollama service is running
        if not OllamaInstaller.check_ollama_running():
            logger.info("Ollama service not running, starting...")
            
            if not await OllamaInstaller.start_ollama_service():
                return False, "Failed to start Ollama service. Please start it manually with 'ollama serve'"
        
        # Check if model is available
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                # Check if our model is in the list
                base_model = model_name.split(':')[0]
                if base_model not in result.stdout:
                    logger.info(f"Model {model_name} not found, pulling...")
                    if not await OllamaInstaller.pull_model(model_name):
                        return False, f"Failed to pull model {model_name}"
            
        except Exception as e:
            logger.warning(f"Could not check model availability: {e}")
        
        return True, "Ollama is ready"


async def setup_ollama(model_name: str) -> bool:
    """
    Convenience function to set up Ollama.
    
    Args:
        model_name: The model to ensure is available
        
    Returns:
        True if Ollama is ready, False otherwise
    """
    success, message = await OllamaInstaller.ensure_ollama_ready(model_name)
    
    if success:
        logger.info(message)
    else:
        logger.error(message)
    
    return success