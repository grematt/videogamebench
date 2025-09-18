from typing import Dict, Any, List, Tuple, Optional
from PIL import Image
import os
import time
import asyncio
import logging
from src.emulators.dos.browser_controller import BrowserController
from src.emulators.interface_base import VideoGameBenchInterface

class DOSGameInterface(VideoGameBenchInterface):
    """DOS Game Interface using Playwright and JSDOS (DOSBOX)"""
    
    def __init__(self, 
                 headless: bool = False,
                 game: str = None,
                 lite: bool = False,
                 key_press_delay: float = 0.1,
                 lite_key_press_delay: float = 0.1,
                 num_screenshots_per_action: int = 0,
                 ):
        super().__init__()
        self.headless = headless
        self.game = game
        self.lite = lite
        self.key_press_delay_ms = key_press_delay * 1000
        self.browser = BrowserController(headless=headless)
        self.num_screenshots_per_action = num_screenshots_per_action

        # Lite condition, can change
        if lite:
            self.key_press_delay_ms = lite_key_press_delay * 1000
            self.num_screenshots_per_action = 3

    async def load_game(self, initial_url: str) -> bool:
        """
        Load a DOS game from a URL.
        """
        pass
        """
        Start the agent by initializing the browser.
        """
        await self.browser.start()
        
        # Navigate to the initial URL
        await self.browser.navigate(initial_url)

        # Pre-loaded actions based on game
        await self.browser.pre_load(self.game)

        if self.lite:
            await self.browser.press_key("Alt+Pause", delay_ms=0)
    
    
    async def click(self, action_input: str, press_key_delay: float = 0.5) -> str:
        x, y = self.browser.current_mouse_position
        click_options = {}
        if action_input:
            if "right" in action_input.lower():
                click_options["button"] = "right"
            
            modifiers = []
            if "shift" in action_input.lower():
                modifiers.append("Shift")
            if "ctrl" in action_input.lower():
                modifiers.append("Control")
            if "alt" in action_input.lower():
                modifiers.append("Alt")
            if modifiers:
                click_options["modifiers"] = modifiers
        else:
            click_options = None
        
        # Click the mouse
        await self.browser.click(x, y, click_options)
        result = f"Mouse clicked at ({x}, {y}) with options: {click_options}"
        return result

    async def move(self, action_input: str, press_key_delay_ms: float = 0.5) -> str:
        x, y = map(float, action_input.split(","))
        await self.browser.move_mouse(x, y)
        result = f"Mouse moved to ({x}, {y})"
        return result

    async def drag(self, action_input: str, press_key_delay_ms: float = 0.5) -> str:
        x, y = map(float, action_input.split(","))
        await self.browser.drag(x, y)
        result = f"Mouse dragged to ({x}, {y})"
        return result

    async def scroll_down(self, action_input: str, press_key_delay_ms: float = 100) -> str:
        amount = int(action_input)
        await self.browser.scroll_down(amount)
        result = f"Scrolled down {amount} pixels."
        return result

    async def scroll_up(self, action_input: str, press_key_delay_ms: float = 100) -> str:
        amount = int(action_input)
        await self.browser.scroll_up(amount)
        result = f"Scrolled up {amount} pixels."
        return result

    async def write(self, action_input: str, press_key_delay_ms: float = 100) -> str:
        await self.browser.type_text(action_input)
        result = f"Typed: {action_input}"
        return result

    async def press_key(self, action_input: str, press_key_delay_ms: float = 100) -> Tuple[str, List[bytes]]:
        screenshots = []
        if "," in action_input:
            keys = action_input.split(",")
            for key in keys:
                await self.browser.press_key(key.strip(), lite_mode=self.lite, delay_ms=press_key_delay_ms)
                screenshot = await self.browser.get_screenshot()
                screenshots.append(screenshot)
                await asyncio.sleep(press_key_delay_ms / 1000)
            result = f"Pressed keys: {action_input}"
        else:
            await self.browser.press_key(action_input, lite_mode=self.lite, delay_ms=press_key_delay_ms)
            result = f"Pressed key: {action_input}"
        return result, screenshots

    async def hold_key(self, action_input: str, delay_ms: float = 100) -> str:
        parts = action_input.split(",")
        key = parts[0]
        duration = float(parts[1]) if len(parts) > 1 else 0.5
        await self.browser.press_key(key, lite_mode=self.lite, delay_ms=duration)
        result = f"Held key {key} for {duration} seconds"
        return result


    async def step(self, 
                    action: str, 
                    action_input: str,
                    key_press_delay_ms: Optional[float] = None,
                    ) -> str:
        """Execute an action and return the observation."""
        key_press_delay_ms = self.key_press_delay_ms if key_press_delay_ms is None else key_press_delay_ms

        try:
            # Execute the action
            result = None
            frames = []

            if self.lite:
                await self.browser.press_key("Alt+Pause", delay_ms=0)
                await asyncio.sleep(0.01)

            action_map = {
                'click': self.click,
                'move': self.move,
                'move_mouse': self.move, # Add alias for move_mouse
                'move_mouse_left': lambda *args: self.move("left", *args),
                'move_mouse_right': lambda *args: self.move("right", *args), 
                'move_mouse_up': lambda *args: self.move("up", *args),
                'move_mouse_down': lambda *args: self.move("down", *args),
                'drag': self.drag,
                'scroll_down': self.scroll_down,
                'scroll_up': self.scroll_up,
                'write': self.write,
                'press_key': self.press_key,
                'hold_key': self.hold_key,
            }

            if action is None:
                result = "No action provided."
            else:
                action = action.lower().strip()
                if action in action_map.keys():
                    result = await action_map[action](action_input, key_press_delay_ms)
                    if isinstance(result, tuple):
                        result, frames = result
                else:
                    result = f"Unknown action: {action}"

            # Take screenshots for approximately 0.5 seconds
            for _ in range(self.num_screenshots_per_action):
                frame = await self.browser.get_screenshot()
                frames.append(frame)
                await asyncio.sleep(key_press_delay_ms / 1000) 

            # Pause game
            if self.lite:
                await self.browser.press_key("Alt+Pause", delay_ms=0)
            
            # Under real benchmark (not lite), take screenshot here
            if not frames or len(frames) == 0:
                screenshot = await self.browser.get_screenshot()
                frames = [screenshot]
                
            return result if result else f"Unknown action: {action}", frames

        except Exception as e:
            error_msg = f"Error executing action: {str(e)}"
            exit()
            
            if self.lite:
                await self.browser.press_key("Alt+Pause", delay_ms=0)
            
            screenshot = await self.browser.get_screenshot()
            return error_msg, [screenshot]
        
    async def close(self) -> None:
        """Clean up resource[screen]."""
        await self.browser.close()

    async def get_observation(self) -> Optional[Dict[str, Any]]:
        """
        Get current screenshot from Playwright.
        """
        screenshot = await self.browser.get_screenshot()
        return screenshot