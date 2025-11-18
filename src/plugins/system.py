"""Plugin system for extensible actions."""

from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
import asyncio


@dataclass
class PluginResult:
    """Result from a plugin execution."""
    success: bool
    data: Any
    message: str
    execution_time: float


class Plugin(ABC):
    """Base class for plugins."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.enabled = True

    @abstractmethod
    async def execute(self, context: Dict, parameters: Dict) -> PluginResult:
        """Execute the plugin."""
        pass

    @abstractmethod
    def can_handle(self, intent: str, entities: Dict) -> bool:
        """Check if plugin can handle the request."""
        pass


class ActionPlugin(Plugin):
    """Plugin that performs actions."""

    def __init__(
        self,
        name: str,
        description: str,
        intents: List[str],
        handler: Callable
    ):
        super().__init__(name, description)
        self.intents = intents
        self.handler = handler

    async def execute(self, context: Dict, parameters: Dict) -> PluginResult:
        """Execute the action."""
        start = datetime.utcnow()
        try:
            if asyncio.iscoroutinefunction(self.handler):
                result = await self.handler(context, parameters)
            else:
                result = self.handler(context, parameters)

            execution_time = (datetime.utcnow() - start).total_seconds()

            return PluginResult(
                success=True,
                data=result,
                message=f"Action '{self.name}' completed successfully",
                execution_time=execution_time
            )
        except Exception as e:
            execution_time = (datetime.utcnow() - start).total_seconds()
            return PluginResult(
                success=False,
                data=None,
                message=f"Action '{self.name}' failed: {str(e)}",
                execution_time=execution_time
            )

    def can_handle(self, intent: str, entities: Dict) -> bool:
        """Check if plugin can handle this intent."""
        return intent in self.intents


class PluginSystem:
    """
    Manages plugins for extensible actions.

    Features:
    - Plugin registration
    - Intent-based routing
    - Execution tracking
    - Plugin chaining
    """

    def __init__(self):
        self.plugins: Dict[str, Plugin] = {}
        self.execution_history: List[Dict] = []
        self.hooks: Dict[str, List[Callable]] = {
            "before_execute": [],
            "after_execute": []
        }

    def register(self, plugin: Plugin):
        """Register a plugin."""
        self.plugins[plugin.name] = plugin

    def unregister(self, plugin_name: str):
        """Unregister a plugin."""
        if plugin_name in self.plugins:
            del self.plugins[plugin_name]

    def get_plugin(self, name: str) -> Optional[Plugin]:
        """Get a plugin by name."""
        return self.plugins.get(name)

    def list_plugins(self) -> List[Dict]:
        """List all registered plugins."""
        return [
            {
                "name": p.name,
                "description": p.description,
                "enabled": p.enabled,
                "type": type(p).__name__
            }
            for p in self.plugins.values()
        ]

    def find_handler(self, intent: str, entities: Dict) -> Optional[Plugin]:
        """Find a plugin that can handle the request."""
        for plugin in self.plugins.values():
            if plugin.enabled and plugin.can_handle(intent, entities):
                return plugin
        return None

    async def execute(
        self,
        plugin_name: str,
        context: Dict,
        parameters: Dict
    ) -> PluginResult:
        """Execute a plugin by name."""
        plugin = self.plugins.get(plugin_name)
        if not plugin:
            return PluginResult(
                success=False,
                data=None,
                message=f"Plugin '{plugin_name}' not found",
                execution_time=0
            )

        if not plugin.enabled:
            return PluginResult(
                success=False,
                data=None,
                message=f"Plugin '{plugin_name}' is disabled",
                execution_time=0
            )

        # Run before hooks
        for hook in self.hooks["before_execute"]:
            await self._run_hook(hook, plugin_name, context, parameters)

        # Execute plugin
        result = await plugin.execute(context, parameters)

        # Run after hooks
        for hook in self.hooks["after_execute"]:
            await self._run_hook(hook, plugin_name, context, result)

        # Log execution
        self.execution_history.append({
            "plugin": plugin_name,
            "timestamp": datetime.utcnow(),
            "success": result.success,
            "execution_time": result.execution_time
        })

        return result

    async def execute_for_intent(
        self,
        intent: str,
        entities: Dict,
        context: Dict,
        parameters: Dict
    ) -> Optional[PluginResult]:
        """Find and execute plugin for an intent."""
        plugin = self.find_handler(intent, entities)
        if not plugin:
            return None

        return await self.execute(plugin.name, context, parameters)

    async def _run_hook(self, hook: Callable, *args):
        """Run a hook function."""
        try:
            if asyncio.iscoroutinefunction(hook):
                await hook(*args)
            else:
                hook(*args)
        except Exception:
            pass  # Hooks should not break execution

    def add_hook(self, event: str, handler: Callable):
        """Add a hook for an event."""
        if event in self.hooks:
            self.hooks[event].append(handler)

    def enable_plugin(self, name: str):
        """Enable a plugin."""
        if name in self.plugins:
            self.plugins[name].enabled = True

    def disable_plugin(self, name: str):
        """Disable a plugin."""
        if name in self.plugins:
            self.plugins[name].enabled = False

    def get_execution_stats(self) -> Dict:
        """Get plugin execution statistics."""
        if not self.execution_history:
            return {"total": 0}

        total = len(self.execution_history)
        successful = sum(1 for e in self.execution_history if e["success"])
        avg_time = statistics.mean([e["execution_time"] for e in self.execution_history])

        by_plugin = {}
        for execution in self.execution_history:
            name = execution["plugin"]
            if name not in by_plugin:
                by_plugin[name] = {"count": 0, "success": 0}
            by_plugin[name]["count"] += 1
            if execution["success"]:
                by_plugin[name]["success"] += 1

        return {
            "total": total,
            "success_rate": successful / total,
            "avg_execution_time": avg_time,
            "by_plugin": by_plugin
        }


# Import for stats
import statistics


# Built-in plugins
def create_builtin_plugins() -> List[Plugin]:
    """Create built-in action plugins."""
    plugins = []

    # Echo plugin (for testing)
    async def echo_handler(context: Dict, parameters: Dict) -> Dict:
        return {"echoed": parameters.get("message", "")}

    plugins.append(ActionPlugin(
        name="echo",
        description="Echo back the input (for testing)",
        intents=["test", "echo"],
        handler=echo_handler
    ))

    # Reminder plugin
    async def reminder_handler(context: Dict, parameters: Dict) -> Dict:
        return {
            "reminder_set": True,
            "message": parameters.get("message", ""),
            "time": parameters.get("time", ""),
            "id": f"reminder_{datetime.utcnow().timestamp()}"
        }

    plugins.append(ActionPlugin(
        name="reminder",
        description="Set reminders",
        intents=["reminder", "remind"],
        handler=reminder_handler
    ))

    # Note plugin
    async def note_handler(context: Dict, parameters: Dict) -> Dict:
        return {
            "note_saved": True,
            "content": parameters.get("content", ""),
            "id": f"note_{datetime.utcnow().timestamp()}"
        }

    plugins.append(ActionPlugin(
        name="notes",
        description="Save notes",
        intents=["note", "save", "remember"],
        handler=note_handler
    ))

    # Search plugin
    async def search_handler(context: Dict, parameters: Dict) -> Dict:
        query = parameters.get("query", "")
        return {
            "searched": True,
            "query": query,
            "results": [],  # Would integrate with memory search
            "message": f"Searched for: {query}"
        }

    plugins.append(ActionPlugin(
        name="search",
        description="Search memories and knowledge",
        intents=["search", "find", "lookup"],
        handler=search_handler
    ))

    return plugins
