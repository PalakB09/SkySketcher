import asyncio
import websockets
import json
import uuid
from datetime import datetime
import base64
import numpy as np
import cv2

class CollaborativeCanvasServer:
    def __init__(self):
        self.clients = {}  # Store connected clients
        self.canvas_state = {
            'width': 1200,
            'height': 720,
            'toolbar_height': 140,
            'canvas_data': None  # Will store base64 encoded canvas
        }
        self.drawing_events = []  # Store drawing events for new clients
        
    async def register_client(self, websocket, path):
        """Register a new client"""
        client_id = str(uuid.uuid4())
        self.clients[client_id] = {
            'websocket': websocket,
            'user_name': f"User_{client_id[:8]}",
            'color': self.get_random_color(),
            'connected_at': datetime.now()
        }
        
        print(f"Client {client_id} connected. Total clients: {len(self.clients)}")
        
        # Send welcome message with client info
        await self.send_to_client(client_id, {
            'type': 'welcome',
            'client_id': client_id,
            'user_name': self.clients[client_id]['user_name'],
            'color': self.clients[client_id]['color'],
            'canvas_state': self.canvas_state
        })
        
        # Send current canvas state to new client
        if self.canvas_state['canvas_data']:
            await self.send_to_client(client_id, {
                'type': 'canvas_sync',
                'canvas_data': self.canvas_state['canvas_data']
            })
        
        # Send existing drawing events
        for event in self.drawing_events[-100:]:  # Send last 100 events
            await self.send_to_client(client_id, event)
        
        # Notify all clients about new user
        await self.broadcast({
            'type': 'user_joined',
            'user_name': self.clients[client_id]['user_name'],
            'client_id': client_id,
            'color': self.clients[client_id]['color'],
            'total_users': len(self.clients)
        }, exclude_client=client_id)
        
        try:
            async for message in websocket:
                await self.handle_message(client_id, message)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister_client(client_id)

    async def unregister_client(self, client_id):
        """Remove a client and notify others"""
        if client_id in self.clients:
            user_name = self.clients[client_id]['user_name']
            del self.clients[client_id]
            print(f"Client {client_id} disconnected. Total clients: {len(self.clients)}")
            
            # Notify remaining clients
            await self.broadcast({
                'type': 'user_left',
                'user_name': user_name,
                'client_id': client_id,
                'total_users': len(self.clients)
            })

    async def handle_message(self, client_id, message):
        """Handle incoming messages from clients"""
        try:
            data = json.loads(message)
            message_type = data.get('type')
            
            if message_type == 'drawing_event':
                await self.handle_drawing_event(client_id, data)
            elif message_type == 'canvas_clear':
                await self.handle_canvas_clear(client_id, data)
            elif message_type == 'canvas_save':
                await self.handle_canvas_save(client_id, data)
            elif message_type == 'diagram_generated':
                await self.handle_diagram_generated(client_id, data)
            elif message_type == 'user_cursor':
                await self.handle_user_cursor(client_id, data)
            elif message_type == 'tool_change':
                await self.handle_tool_change(client_id, data)
                
        except json.JSONDecodeError:
            print(f"Invalid JSON from client {client_id}")
        except Exception as e:
            print(f"Error handling message from {client_id}: {e}")

    async def handle_drawing_event(self, client_id, data):
        """Handle drawing events and broadcast to other clients"""
        # Add user info to the drawing event
        if client_id in self.clients:
            data['user_name'] = self.clients[client_id]['user_name']
            data['client_id'] = client_id
            data['timestamp'] = datetime.now().isoformat()
            
            # Store event for new clients
            self.drawing_events.append(data)
            if len(self.drawing_events) > 1000:  # Keep only last 1000 events
                self.drawing_events = self.drawing_events[-1000:]
            
            # Broadcast to all other clients
            await self.broadcast(data, exclude_client=client_id)

    async def handle_canvas_clear(self, client_id, data):
        """Handle canvas clear and broadcast to other clients"""
        if client_id in self.clients:
            data['user_name'] = self.clients[client_id]['user_name']
            data['client_id'] = client_id
            data['timestamp'] = datetime.now().isoformat()
            
            # Clear stored events
            self.drawing_events = []
            self.canvas_state['canvas_data'] = None
            
            # Broadcast to all clients
            await self.broadcast(data)

    async def handle_canvas_save(self, client_id, data):
        """Handle canvas save event"""
        if client_id in self.clients:
            # Update stored canvas state
            if 'canvas_data' in data:
                self.canvas_state['canvas_data'] = data['canvas_data']
            
            data['user_name'] = self.clients[client_id]['user_name']
            data['client_id'] = client_id
            data['timestamp'] = datetime.now().isoformat()
            
            # Broadcast save notification to all clients
            await self.broadcast({
                'type': 'canvas_saved',
                'user_name': data['user_name'],
                'timestamp': data['timestamp']
            })

    async def handle_diagram_generated(self, client_id, data):
        """Handle diagram generation and share with all clients"""
        if client_id in self.clients:
            data['user_name'] = self.clients[client_id]['user_name']
            data['client_id'] = client_id
            data['timestamp'] = datetime.now().isoformat()
            
            # Broadcast generated diagram to all clients
            await self.broadcast(data)

    async def handle_user_cursor(self, client_id, data):
        """Handle user cursor position updates"""
        if client_id in self.clients:
            data['user_name'] = self.clients[client_id]['user_name']
            data['client_id'] = client_id
            data['color'] = self.clients[client_id]['color']
            
            # Broadcast cursor position to other clients
            await self.broadcast(data, exclude_client=client_id)

    async def handle_tool_change(self, client_id, data):
        """Handle tool/color changes"""
        if client_id in self.clients:
            data['user_name'] = self.clients[client_id]['user_name']
            data['client_id'] = client_id
            
            # Broadcast tool change to other clients
            await self.broadcast(data, exclude_client=client_id)

    async def send_to_client(self, client_id, data):
        """Send message to a specific client"""
        if client_id in self.clients:
            try:
                await self.clients[client_id]['websocket'].send(json.dumps(data))
            except websockets.exceptions.ConnectionClosed:
                await self.unregister_client(client_id)

    async def broadcast(self, data, exclude_client=None):
        """Broadcast message to all clients except excluded one"""
        if not self.clients:
            return
            
        message = json.dumps(data)
        disconnected_clients = []
        
        for client_id, client_info in self.clients.items():
            if client_id != exclude_client:
                try:
                    await client_info['websocket'].send(message)
                except websockets.exceptions.ConnectionClosed:
                    disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            await self.unregister_client(client_id)

    def get_random_color(self):
        """Generate a random color for user identification"""
        colors = [
            [255, 0, 0],    # Red
            [0, 255, 0],    # Green
            [0, 0, 255],    # Blue
            [255, 255, 0],  # Yellow
            [255, 0, 255],  # Magenta
            [0, 255, 255],  # Cyan
            [255, 165, 0],  # Orange
            [128, 0, 128],  # Purple
            [255, 192, 203], # Pink
            [0, 128, 0],    # Dark Green
        ]
        import random
        return colors[random.randint(0, len(colors) - 1)]

# Server setup
async def main():
    server = CollaborativeCanvasServer()
    print("Starting collaborative canvas server on ws://localhost:8765")
    print("Press Ctrl+C to stop the server")
    
    async with websockets.serve(server.register_client, "localhost", 8765):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer stopped") 