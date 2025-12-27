"""
LLM Configuration Loader
Loads and initializes LLMs from config/llm.json
"""
import json
import os
import importlib
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)


class LLMConfigLoader:
    """Loads and manages LLM configurations from llm.json"""
    
    def __init__(self, config_path: str = "config/llm.json"):
        """
        Initialize the LLM Config Loader
        
        Args:
            config_path: Path to the llm.json configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.llm_instances = {}
        
        # Load LLMs marked for initialization
        self._load_initial_llms()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load the JSON configuration file"""
        with open(self.config_path, 'r') as f:
            return json.load(f)
    
    def _resolve_env_value(self, value: str) -> str:
        """
        Resolve environment variable references in config values
        
        Args:
            value: Configuration value (may contain ENV:VARIABLE_NAME)
            
        Returns:
            Resolved value from environment or original value
        """
        if isinstance(value, str) and value.startswith("ENV:"):
            env_var = value.split("ENV:")[1]
            env_value = os.getenv(env_var)
            if env_value is None:
                raise ValueError(f"Environment variable {env_var} not found")
            return env_value
        return value
    
    def _resolve_config_values(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively resolve all ENV: references in config dictionary
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Resolved configuration dictionary
        """
        resolved = {}
        for key, value in config.items():
            if isinstance(value, dict):
                resolved[key] = self._resolve_config_values(value)
            elif isinstance(value, str):
                resolved[key] = self._resolve_env_value(value)
            else:
                resolved[key] = value
        return resolved
    
    def _load_initial_llms(self):
        """Load LLMs that have load_on_init set to True"""
        llms_config = self.config.get("llms", {})
        
        for llm_name, llm_config in llms_config.items():
            if llm_config.get("load_on_init", "False").lower() == "true":
                try:
                    self.load_llm(llm_name)
                    print(f"✓ Loaded LLM: {llm_name}")
                except Exception as e:
                    print(f"✗ Failed to load LLM {llm_name}: {str(e)}")
    
    def load_llm(self, llm_name: str):
        """
        Load and instantiate an LLM by name
        
        Args:
            llm_name: Name of the LLM from configuration
            
        Returns:
            Instantiated LLM object
        """
        if llm_name in self.llm_instances:
            return self.llm_instances[llm_name]
        
        llms_config = self.config.get("llms", {})
        if llm_name not in llms_config:
            raise ValueError(f"LLM '{llm_name}' not found in configuration")
        
        llm_config = llms_config[llm_name]
        
        # Get import details
        module_name = llm_config.get("import_module")
        class_name = llm_config.get("import_class")
        
        if not module_name or not class_name:
            raise ValueError(f"Missing import_module or import_class for {llm_name}")
        
        # Import the module and class
        try:
            module = importlib.import_module(module_name)
            llm_class = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Could not import {class_name} from {module_name}: {str(e)}")
        
        # Resolve configuration values (ENV references)
        resolved_config = self._resolve_config_values(llm_config.get("config", {}))
        
        # Instantiate the LLM
        llm_instance = llm_class(**resolved_config)
        
        # Cache the instance
        self.llm_instances[llm_name] = llm_instance
        
        return llm_instance
    
    def get_manager_llm(self):
        """
        Get the manager LLM instance
        
        Returns:
            The configured manager LLM instance
        """
        manager_llm_name = self.config.get("managerLLM")
        if not manager_llm_name:
            raise ValueError("No managerLLM specified in configuration")
        
        return self.load_llm(manager_llm_name)
    
    def get_llm_config(self, llm_name: str) -> Dict[str, Any]:
        """
        Get the configuration for a specific LLM
        
        Args:
            llm_name: Name of the LLM
            
        Returns:
            Configuration dictionary for the LLM
        """
        llms_config = self.config.get("llms", {})
        if llm_name not in llms_config:
            raise ValueError(f"LLM '{llm_name}' not found in configuration")
        
        return llms_config[llm_name]
    
    def list_available_llms(self) -> list:
        """
        List all available LLMs in the configuration
        
        Returns:
            List of LLM names
        """
        return list(self.config.get("llms", {}).keys())


# Global instance for easy access
_llm_loader: Optional[LLMConfigLoader] = None


def get_llm_loader() -> LLMConfigLoader:
    """
    Get the global LLM loader instance (singleton pattern)
    
    Returns:
        LLMConfigLoader instance
    """
    global _llm_loader
    if _llm_loader is None:
        _llm_loader = LLMConfigLoader()
    return _llm_loader


def get_manager_llm():
    """
    Convenience function to get the manager LLM
    
    Returns:
        Manager LLM instance
    """
    return get_llm_loader().get_manager_llm()
