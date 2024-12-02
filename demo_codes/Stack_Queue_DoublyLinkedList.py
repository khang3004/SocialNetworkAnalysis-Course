class StackUnderflowError(Exception):
    def __init__(self, message="Stack is empty! Cannot pop element."):
        self.message = message
        super().__init__(self.message)

class QueueUnderflowError(Exception):
    def __init__(self, message="Queue is empty! Cannot dequeue element."):
        self.message = message
        super().__init__(self.message)

class SinglyNode:
    def __init__(self, data):
        self.data = data
        self.next = None

class SinglyLinkedList:
    def __init__(self):
        self.head = None
        self._size = 0  # Initialize size

    def is_empty(self):
        """Check if the list is empty."""
        return self.head is None
    
    def display(self):
        """Display all elements in the linked list."""
        if self.is_empty():
            print("The list is empty.")
            return
        current = self.head
        while current:
            if current.next:  
                print(f"{current.data} ->", end=" ")
            else:
                print(current.data, end=" ")
            current = current.next
        print()

    def insert_at_front(self, data):
        """Insert a new node at the front of the list."""
        new_node = SinglyNode(data)
        new_node.next = self.head
        self.head = new_node
        self._size += 1

    def insert_at_end(self, data):
        """Insert a new node at the end of the list."""
        new_node = SinglyNode(data)
        if self.is_empty():
            self.head = new_node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node
        self._size += 1

    def delete_from_front(self):
        """Delete the node at the front of the list."""
        if self.is_empty():
            print("Cannot delete from an empty list.")
            return None
        data = self.head.data
        self.head = self.head.next
        self._size -= 1
        return data

    def delete_from_end(self):
        """Delete the node at the end of the list."""
        if self.is_empty():
            print("Cannot delete from an empty list.")
            return None
        current = self.head
        if self.head.next is None:
            data = self.head.data
            self.head = None
        else:
            while current.next.next:
                current = current.next
            data = current.next.data
            current.next = None
        self._size -= 1
        return data

    def delete(self, data):
        """Delete the first occurrence of data in the list."""
        current = self.head
        prev = None
        while current:
            if current.data == data:
                if prev is None:
                    self.head = current.next  # Remove head
                else:
                    prev.next = current.next  # Bypass the node
                self._size -= 1
                print(f"Deleted: {data}")
                return
            prev = current
            current = current.next
        print(f"{data} not found in the list.")

    def search(self, data):
        """Search for the first occurrence of data in the list."""
        current = self.head
        position = 0
        while current:
            if current.data == data:
                print(f"Found {data} at position {position}")
                return current
            current = current.next
            position += 1
        print(f"{data} not found in the list.")
        return None

    def peek_front(self):
        """Return the data at the front of the list without removing it."""
        if self.is_empty():
            return None
        return self.head.data

    def peek_end(self):
        """Return the data at the end of the list without removing it."""
        if self.is_empty():
            return None
        current = self.head
        while current.next:
            current = current.next
        return current.data

    def size(self):
        """Return the number of nodes in the list."""
        return self._size

    def __len__(self):
        """Return the size of the list."""
        return self._size

    def __str__(self):
        """Return a string representation of the list."""
        if self.is_empty():
            return "The list is empty."
        result = []
        current = self.head
        while current:
            result.append(str(current.data))
            current = current.next
        return " -> ".join(result)

    def __contains__(self, item):
        """Check if the item is in the list."""
        return self.search(item) is not None

    def __iter__(self):
        """Iterate over the list."""
        current = self.head
        while current:
            yield current.data
            current = current.next

    def clear(self):
        """Clear all nodes from the list."""
        self.head = None
        self._size = 0
        print("The list has been cleared.")

    def display_reverse(self):
        """Display all elements in reverse order using recursion."""
        def _reverse_print(node):
            if node:
                _reverse_print(node.next)
                print(node.data, end=" <- " if node != self.head else "\n")

        if self.is_empty():
            print("The list is empty.")
        else:
            _reverse_print(self.head)


    @staticmethod
    def is_palindrome(string):
        return string == string[::-1]

    def insert_floor(self, floor):
        new_node = SinglyNode(floor)
        if self.head is None or self.head.data > floor:
            new_node.next = self.head
            self.head = new_node
        else:
            current = self.head
            while current.next and current.next.data < floor:
                current = current.next
            new_node.next = current.next
            current.next = new_node
        self._size += 1

    def sort_ascending(self):
        if self.is_empty() or self.head.next is None:
            return
        sorted = False
        while not sorted:
            sorted = True
            current = self.head
            while current and current.next:
                if current.data > current.next.data:
                    current.data, current.next.data = current.next.data, current.data
                    sorted = False
                current = current.next

    def sort_descending(self):
        if self.is_empty() or self.head.next is None:
            return
        sorted = False
        while not sorted:
            sorted = True
            current = self.head
            while current and current.next:
                if current.data < current.next.data:
                    current.data, current.next.data = current.next.data, current.data
                    sorted = False
                current = current.next

    def display_floors(self):
        current = self.head
        while current:
            print(f"Arriving at floor: {current.data}")
            current = current.next

    def display_reverse(self):
        """Hiển thị danh sách theo thứ tự ngược"""
        def _reverse_print(node):
            if node:
                _reverse_print(node.next)
                print(node.data, end=" <- " if node != self.head else "\n")

        if self.is_empty():
            print("The list is empty.")
        else:
            _reverse_print(self.head)



class Node:
    def __init__(self, data):
        self.data = data
        self.prev = None
        self.next = None

class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self._size = 0  # Initialize size

    def is_empty(self):
        return self.head is None
    
    def display(self):
        if self.is_empty():
            print("The list is empty.")
            return
        current = self.head
        while current:
            if current.next:  
                print(f"{current.data} <->", end=" ")
            else:
                print(current.data, end=" ")
            current = current.next
        print()

    def insert_at_front(self, data):
        new_node = Node(data)
        if self.is_empty():
            self.head = self.tail = new_node
        else:
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node
        self._size += 1  # Increment size when inserting

    def insertAtPosition(self, position, data):
        """Chèn một phần tử vào vị trí chỉ định trong danh sách liên kết đôi"""
        if position < 0 or position > self._size:
            print("Vị trí không hợp lệ")
            return
        
        if position == 0:
            self.insert_at_front(data)
            return
            
        if position == self._size:
            self.insert_at_end(data)
            return
            
        new_node = Node(data)
        current = self.head
        for _ in range(position - 1):
            current = current.next
            
        new_node.next = current.next
        new_node.prev = current
        current.next.prev = new_node
        current.next = new_node
        self._size += 1

    def insert_at_end(self, data):
        new_node = Node(data)
        if self.is_empty():
            self.head = self.tail = new_node
        else:
            new_node.prev = self.tail
            self.tail.next = new_node
            self.tail = new_node
        self._size += 1  # Increment size when inserting

    def delete_from_front(self):
        if self.is_empty():
            return None
        data = self.head.data
        self.head = self.head.next
        if self.head is not None:
            self.head.prev = None
        else:
            self.tail = None  # List becomes empty
        self._size -= 1  # Decrement size when deleting
        return data

    def delete_from_end(self):
        if self.is_empty():
            return None
        data = self.tail.data
        self.tail = self.tail.prev
        if self.tail is not None:
            self.tail.next = None
        else:
            self.head = None  # List becomes empty
        self._size -= 1  # Decrement size when deleting
        return data

    def delete_at_position(self, position):
        """Delete node at given position (0-based index)."""
        if position < 0 or position >= self._size:
            return None
            
        if position == 0:
            return self.delete_from_front()
            
        if position == self._size - 1:
            return self.delete_from_end()
            
        current = self.head
        for _ in range(position):
            current = current.next
            
        current.prev.next = current.next
        current.next.prev = current.prev
        self._size -= 1
        return current.data

    def peek_front(self):
        if self.is_empty():
            return None
        return self.head.data

    def peek_end(self):
        if self.is_empty():
            return None
        return self.tail.data

    def size(self):
        """Return the size of the doubly linked list."""
        return self._size

    # Magic methods for better usability

    def __len__(self):
        """Return the length of the doubly linked list."""
        return self._size

    def __str__(self):
        """Return a string representation of the list."""
        if self.is_empty():
            return "The list is empty."
        result = []
        current = self.head
        while current:
            result.append(str(current.data))
            current = current.next
        return " <-> ".join(result)

    def __iter__(self):
        """Return an iterator for the doubly linked list."""
        current = self.head
        while current:
            yield current.data
            current = current.next

    def __reversed__(self):
        """Return a reversed iterator for the doubly linked list."""
        current = self.tail
        while current:
            yield current.data
            current = current.prev

    def __contains__(self, item):
        """Check if the item is in the doubly linked list."""
        current = self.head
        while current:
            if current.data == item:
                return True
            current = current.next
        return False

    # Static method for utility functions

    @staticmethod
    def is_palindrome(string):
        """Check if a given string is a palindrome."""
        return string == string[::-1]

    
    def insert_floor(self, floor):
        """Chèn tầng theo thứ tự tăng dần"""
        new_node = Node(floor)
        if self.head is None or self.head.data > floor:
            new_node.next = self.head
            if self.head is not None:
                self.head.prev = new_node
            self.head = new_node
            if self.tail is None:  # Nếu đây là phần tử đầu tiên
                self.tail = new_node
        else:
            current = self.head
            while current.next and current.next.data < floor:
                current = current.next
            new_node.next = current.next
            if current.next is not None:
                current.next.prev = new_node
            else:
                self.tail = new_node  # Cập nhật con trỏ tail nếu chèn vào cuối
            new_node.prev = current
            current.next = new_node

    def sort_ascending(self):
        """Sắp xếp danh sách theo thứ tự tăng dần"""
        if not self.head or not self.head.next:
            return
        sorted = False
        while not sorted:
            sorted = True
            current = self.head
            while current and current.next:
                if current.data > current.next.data:
                    current.data, current.next.data = current.next.data, current.data
                    sorted = False
                current = current.next

    def sort_descending(self):
        """Sắp xếp danh sách theo thứ tự giảm dần"""
        if not self.head or not self.head.next:
            return
        sorted = False
        while not sorted:
            sorted = True
            current = self.head
            while current and current.next:
                if current.data < current.next.data:
                    current.data, current.next.data = current.next.data, current.data
                    sorted = False
                current = current.next

    def display_floors(self):
        """Hiển thị các tầng hiện có trong danh sách"""
        current = self.head
        while current:
            print(f"Arriving at floor: {current.data}")
            current = current.next
    
    def display_reverse(self):
        if self.is_empty():
            print("EMPTY!")
            return
        current = self.tail
        while current:
            if current.prev: 
                print(f"{current.data} <->", end=" ")
            else:
                print(current.data, end=" ")
            current = current.prev

class Stack(DoublyLinkedList):
    def __init__(self):
        super().__init__()

    def push(self, data):
        self.insert_at_front(data)
        # print(f"Pushed {data} onto the stack")

    def pop(self):
        data = self.delete_from_front()
        if data is not None: 
            # print(f"Popped {data} from the stack")
            return data
        raise StackUnderflowError()
        

    def peek(self):
        data = self.peek_front()
        if data is not None: 
            # print(f"Top of the stack is {data}")
            return data
        # print("Stack is empty")
        

class Queue(DoublyLinkedList):
    def __init__(self):
        super().__init__()  
    def enqueue(self, new_data):
        self.insert_at_end(new_data)  # Thêm vào cuối danh sách thay vì đầu
        # print(f"Enqueued {new_data} to the queue ")

    def dequeue(self):
        data_front = self.delete_from_front()
        if data_front is not None:
            # print(f"Dequeued {data_front} from the queue!")
            return data_front
        raise QueueUnderflowError()

    def peek(self):
        data = self.peek_front()
        if data is not None: 
            # print(f"Front of the queue is {data}")
            return data
        # print("Queue is empty")

class PriorityQueueDoublyLinkedList(Queue):
    def __init__(self):
        super().__init__()

    # Override the enqueue method to insert elements based on priority
    def enqueue(self, data, priority):
        """
        Add an element to the queue based on priority.
        Elements with a higher priority (lower number) are placed before others.
        """
        new_node = Node((data, priority))  # Create a node with data as a tuple (data, priority)

        # If the queue is empty or the new element has a higher priority than the first element
        if self.is_empty() or self.head.data[1] > priority:
            # Insert at the beginning of the list
            new_node.next = self.head
            if self.head is not None:
                self.head.prev = new_node
            self.head = new_node
            if self.tail is None:  # If the queue was initially empty
                self.tail = new_node
        else:
            # Insert the element at the appropriate position based on priority
            current = self.head
            while current.next and current.next.data[1] <= priority:
                current = current.next
            new_node.next = current.next
            if current.next is not None:
                current.next.prev = new_node
            new_node.prev = current
            current.next = new_node
            if new_node.next is None:
                self.tail = new_node  # Update the tail if inserted at the end of the list

        print(f"Element {data} with priority {priority} has been added to the queue.")

    # Override the dequeue method to remove the highest priority element
    def dequeue(self):
        """Process and remove the highest priority element (front of the queue)"""
        if self.is_empty():
            raise IndexError("Dequeue from an empty priority queue")
        # Use the dequeue method from the parent class
        data = super().dequeue()
        print(f"Dequeued element: {data[0]} with priority {data[1]}")
        return data

    def peek(self):
        """Peek at the element with the highest priority (the front of the queue)"""
        if self.is_empty():
            raise IndexError("Peek from an empty priority queue")
        return self.peek_front()

    def display(self):
        """Display the current elements in the PriorityQueue."""
        if self.is_empty():
            print("The priority queue is empty.")
            return
        current = self.head
        while current:
            print(f"Element {current.data[0]} with priority {current.data[1]}")
            current = current.next

import heapq

class PriorityQueue:
    def __init__(self):
        self.elements = []
    
    def is_empty(self):
        return len(self.elements) == 0
    
    def enqueue(self, item, priority):
        """
        Add an item with priority
        item: có thể là vertex hoặc tuple (u,v)
        priority: có thể là distance hoặc weight
        """
        heapq.heappush(self.elements, (priority, item))
    
    def dequeue(self):
        """Return (item, priority)"""
        if self.is_empty():
            raise IndexError("Priority queue is empty")
        priority, item = heapq.heappop(self.elements)
        return item, priority  # Đổi thứ tự trả về để phù hợp với cách sử dụng
    
    def peek(self):
        """Return (item, priority) without removing"""
        if self.is_empty():
            raise IndexError("Priority queue is empty")
        return self.elements[0][1], self.elements[0][0]


        