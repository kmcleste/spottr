import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path


class LogGenerator:
    """Generates realistic sample logs for testing purposes."""

    def __init__(self):
        self.components = [
            "UserService",
            "AuthenticationManager",
            "DatabasePool",
            "PaymentProcessor",
            "EmailService",
            "CacheManager",
            "APIGateway",
            "OrderService",
            "InventoryService",
            "NotificationService",
            "WebServer",
            "BackgroundWorker",
        ]

        self.users = [
            "user123",
            "admin",
            "john.doe",
            "alice.smith",
            "bob.wilson",
            "sarah.johnson",
            "mike.davis",
            "lisa.brown",
            "tom.miller",
            "jane.garcia",
        ]

        self.ip_addresses = [
            "192.168.1.10",
            "10.0.0.15",
            "172.16.0.20",
            "203.0.113.5",
            "198.51.100.8",
            "127.0.0.1",
            "10.10.10.100",
            "192.168.0.50",
        ]

        self.error_messages = [
            "Connection timeout after 30 seconds",
            "Failed to authenticate user credentials",
            "Database query execution failed",
            "OutOfMemoryError: Java heap space exhausted",
            "Unable to connect to external service",
            "Payment processing failed - invalid card",
            "Session expired for user",
            "Network unreachable - retrying connection",
            "Slow query detected: execution time 5000ms",
            "Thread pool exhausted - unable to process request",
        ]

        self.normal_messages = [
            "User login successful",
            "Order processed successfully",
            "Cache hit for product data",
            "Email notification sent",
            "Background job completed",
            "Health check passed",
            "Configuration loaded successfully",
            "Database connection established",
            "Request processed in 150ms",
            "User session created",
        ]

    def generate_timestamp(self, base_time: datetime, offset_minutes: int = 0) -> str:
        """Generate a timestamp string."""
        timestamp = base_time + timedelta(
            minutes=offset_minutes,
            seconds=random.randint(0, 59),
            microseconds=random.randint(0, 999999),
        )
        return timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # Include milliseconds

    def generate_normal_log(self, timestamp: str, component: str = None) -> str:
        """Generate a normal log entry."""
        if not component:
            component = random.choice(self.components)

        level = random.choices(["INFO", "DEBUG"], weights=[0.8, 0.2])[0]
        message = random.choice(self.normal_messages)

        # Sometimes add user context
        if random.random() < 0.3:
            user = random.choice(self.users)
            message = f"{message} for user {user}"

        # Sometimes add IP context
        if random.random() < 0.2:
            ip = random.choice(self.ip_addresses)
            message = f"{message} from {ip}"

        return f"{timestamp} [{component}] {level}: {message}"

    def generate_error_log(
        self, timestamp: str, component: str = None, severity: str = "ERROR"
    ) -> str:
        """Generate an error log entry."""
        if not component:
            component = random.choice(self.components)

        message = random.choice(self.error_messages)

        # Add context based on error type
        if "authenticate" in message.lower():
            user = random.choice(self.users)
            ip = random.choice(self.ip_addresses)
            message = f"{message} for user '{user}' from {ip}"
        elif "payment" in message.lower():
            order_id = f"ORDER-{random.randint(10000, 99999)}"
            message = f"{message} for {order_id}"
        elif "query" in message.lower():
            table = random.choice(["users", "orders", "products", "inventory"])
            message = f"{message} on table '{table}'"

        return f"{timestamp} [{component}] {severity}: {message}"

    def generate_performance_issue(self, timestamp: str) -> str:
        """Generate performance-related log entries."""
        scenarios = [
            f"{timestamp} [DatabasePool] WARN: Slow query detected: SELECT * FROM orders WHERE created_at > '2024-01-01' - execution time: {random.randint(3000, 8000)}ms",
            f"{timestamp} [WebServer] WARN: Request processing time exceeded threshold: {random.randint(2000, 5000)}ms for /api/users/search",
            f"{timestamp} [CacheManager] INFO: Cache miss rate increased to {random.randint(40, 80)}% in last 5 minutes",
            f"{timestamp} [BackgroundWorker] WARN: Thread pool utilization at {random.randint(85, 98)}% - consider scaling",
        ]
        return random.choice(scenarios)

    def generate_security_incident(self, timestamp: str) -> str:
        """Generate security-related log entries."""
        suspicious_ips = [
            "203.0.113.666",
            "198.51.100.999",
            "45.33.32.156",
            "185.220.101.23",
        ]

        scenarios = [
            f"{timestamp} [AuthenticationManager] ERROR: Authentication failed for user '{random.choice(self.users)}' from {random.choice(suspicious_ips)} - invalid credentials",
            f"{timestamp} [APIGateway] WARN: Suspicious activity detected: {random.randint(10, 50)} failed login attempts from {random.choice(suspicious_ips)} in 60 seconds",
            f"{timestamp} [WebServer] ERROR: Access denied to /admin endpoint from unauthorized IP {random.choice(suspicious_ips)}",
            f"{timestamp} [AuthenticationManager] ERROR: Brute force attack detected from {random.choice(suspicious_ips)} - blocking IP",
        ]
        return random.choice(scenarios)

    def generate_resource_exhaustion(self, timestamp: str) -> str:
        """Generate resource exhaustion scenarios."""
        scenarios = [
            f"{timestamp} [PaymentProcessor] FATAL: OutOfMemoryError: Java heap space - unable to allocate 1048576 bytes",
            f"{timestamp} [DatabasePool] CRITICAL: Connection pool exhausted - all 50 connections in use",
            f"{timestamp} [WebServer] ERROR: Unable to create new thread - system resource exhausted",
            f"{timestamp} [FileHandler] ERROR: Disk space exhausted on /var/log partition - 0 bytes available",
            f"{timestamp} [BackgroundWorker] FATAL: java.lang.OutOfMemoryError: GC overhead limit exceeded",
        ]
        return random.choice(scenarios)

    def generate_application_log(
        self,
        filename: str,
        duration_hours: int = 2,
        error_rate: float = 0.05,
        include_incidents: bool = True,
    ):
        """Generate a complete application log file."""
        base_time = datetime.now() - timedelta(hours=duration_hours)
        logs = []

        # Calculate total entries (roughly 1 log per second on average)
        total_entries = duration_hours * 3600
        error_entries = int(total_entries * error_rate)
        normal_entries = total_entries - error_entries  # noqa: F841

        print(f"Generating {total_entries} log entries over {duration_hours} hours...")

        # Generate time-distributed entries
        for i in range(total_entries):
            offset_minutes = (i / total_entries) * (duration_hours * 60)
            timestamp = self.generate_timestamp(base_time, int(offset_minutes))

            # Determine log type
            if i < error_entries:
                if random.random() < 0.3:  # 30% chance of severe errors
                    log_entry = self.generate_error_log(timestamp, severity="FATAL")
                else:
                    log_entry = self.generate_error_log(timestamp)
            else:
                log_entry = self.generate_normal_log(timestamp)

            logs.append(log_entry)

        # Add specific incident scenarios if requested
        if include_incidents:
            print("Adding incident scenarios...")

            # Memory exhaustion incident (cluster of related errors)
            incident_time = base_time + timedelta(minutes=30)
            for j in range(5):
                timestamp = self.generate_timestamp(incident_time, j)
                logs.append(self.generate_resource_exhaustion(timestamp))

            # Performance degradation period
            perf_start = base_time + timedelta(minutes=60)
            for j in range(15):
                timestamp = self.generate_timestamp(perf_start, j)
                logs.append(self.generate_performance_issue(timestamp))

            # Security incident
            security_start = base_time + timedelta(minutes=90)
            for j in range(8):
                timestamp = self.generate_timestamp(security_start, j)
                logs.append(self.generate_security_incident(timestamp))

        # Sort logs by timestamp and write to file
        logs.sort()

        with open(filename, "w") as f:
            for log in logs:
                f.write(log + "\n")

        print(f"Generated log file: {filename} ({len(logs)} entries)")

    def generate_high_error_rate_log(self, filename: str, duration_minutes: int = 30):
        """Generate a log file with a high error rate scenario."""
        base_time = datetime.now() - timedelta(minutes=duration_minutes)
        logs = []

        print(f"Generating high error rate scenario over {duration_minutes} minutes...")

        # First 10 minutes: normal operation
        for i in range(600):  # 10 minutes * 60 seconds
            timestamp = self.generate_timestamp(base_time, i // 60)
            logs.append(self.generate_normal_log(timestamp))

        # Next 10 minutes: high error rate (50% errors)
        for i in range(600):
            offset = 10 + (i // 60)
            timestamp = self.generate_timestamp(base_time, offset)
            if i % 2 == 0:  # 50% error rate
                logs.append(self.generate_error_log(timestamp))
            else:
                logs.append(self.generate_normal_log(timestamp))

        # Last 10 minutes: recovery (back to normal)
        for i in range(600):
            offset = 20 + (i // 60)
            timestamp = self.generate_timestamp(base_time, offset)
            if random.random() < 0.05:  # 5% error rate
                logs.append(self.generate_error_log(timestamp))
            else:
                logs.append(self.generate_normal_log(timestamp))

        # Sort and write
        logs.sort()
        with open(filename, "w") as f:
            for log in logs:
                f.write(log + "\n")

        print(f"Generated high error rate log: {filename} ({len(logs)} entries)")

    def generate_microservice_logs(self, base_filename: str, num_services: int = 3):
        """Generate separate log files for different microservices."""
        services = [
            "user-service",
            "payment-service",
            "inventory-service",
            "notification-service",
        ]
        selected_services = services[:num_services]

        for service in selected_services:
            filename = f"{base_filename}_{service}.log"
            # Focus each service on specific types of logs
            if "user" in service:
                self._generate_user_service_logs(filename)
            elif "payment" in service:
                self._generate_payment_service_logs(filename)
            elif "inventory" in service:
                self._generate_inventory_service_logs(filename)
            else:
                self.generate_application_log(
                    filename, duration_hours=1, error_rate=0.03
                )

    def _generate_user_service_logs(self, filename: str):
        """Generate user service specific logs with auth patterns."""
        base_time = datetime.now() - timedelta(hours=1)
        logs = []

        # Normal user operations
        for i in range(500):
            timestamp = self.generate_timestamp(base_time, i // 10)
            user = random.choice(self.users)
            ip = random.choice(self.ip_addresses)

            operations = [
                f"User {user} logged in successfully from {ip}",
                f"User profile updated for {user}",
                f"Password reset requested for {user}",
                f"User {user} logged out",
                f"Session validated for {user}",
            ]

            message = random.choice(operations)
            level = "INFO"

            # Add some auth failures
            if random.random() < 0.08:
                message = f"Authentication failed for user {user} from {ip} - invalid password"
                level = "ERROR"

            logs.append(f"{timestamp} [UserService] {level}: {message}")

        # Simulate brute force attack
        attack_time = base_time + timedelta(minutes=30)
        attacker_ip = "203.0.113.666"
        for i in range(20):
            timestamp = self.generate_timestamp(attack_time, i // 4)
            target_user = random.choice(self.users[:3])  # Focus on few users
            logs.append(
                f"{timestamp} [UserService] ERROR: Authentication failed for user {target_user} from {attacker_ip} - invalid credentials"
            )

        logs.sort()
        with open(filename, "w") as f:
            for log in logs:
                f.write(log + "\n")

        print(f"Generated user service log: {filename}")

    def _generate_payment_service_logs(self, filename: str):
        """Generate payment service logs with transaction patterns."""
        base_time = datetime.now() - timedelta(hours=1)
        logs = []

        for i in range(400):
            timestamp = self.generate_timestamp(base_time, i // 8)
            order_id = f"ORDER-{random.randint(10000, 99999)}"
            amount = random.randint(10, 500)

            # Most transactions succeed
            if random.random() < 0.9:
                logs.append(
                    f"{timestamp} [PaymentProcessor] INFO: Payment processed successfully for {order_id} - amount: ${amount}"
                )
            else:
                error_reasons = [
                    "insufficient funds",
                    "invalid card number",
                    "card expired",
                    "payment gateway timeout",
                    "fraud detection triggered",
                ]
                reason = random.choice(error_reasons)
                logs.append(
                    f"{timestamp} [PaymentProcessor] ERROR: Payment failed for {order_id} - {reason}"
                )

        # Add some memory issues during peak load
        peak_time = base_time + timedelta(minutes=45)
        for i in range(3):
            timestamp = self.generate_timestamp(peak_time, i)
            logs.append(
                f"{timestamp} [PaymentProcessor] FATAL: OutOfMemoryError during payment processing - high transaction volume"
            )

        logs.sort()
        with open(filename, "w") as f:
            for log in logs:
                f.write(log + "\n")

        print(f"Generated payment service log: {filename}")

    def _generate_inventory_service_logs(self, filename: str):
        """Generate inventory service logs with stock management patterns."""
        base_time = datetime.now() - timedelta(hours=1)
        logs = []

        products = ["PROD-001", "PROD-002", "PROD-003", "PROD-004", "PROD-005"]

        for i in range(300):
            timestamp = self.generate_timestamp(base_time, i // 6)
            product = random.choice(products)

            operations = [
                f"Stock updated for {product} - new quantity: {random.randint(0, 100)}",
                f"Stock check performed for {product}",
                f"Inventory sync completed for {product}",
                f"Reorder point reached for {product}",
                f"Stock reservation created for {product}",
            ]

            if random.random() < 0.95:
                message = random.choice(operations)
                level = "INFO"
            else:
                message = (
                    f"Database connection timeout during stock update for {product}"
                )
                level = "ERROR"

            logs.append(f"{timestamp} [InventoryService] {level}: {message}")

        # Add slow query issues
        slow_query_time = base_time + timedelta(minutes=20)
        for i in range(5):
            timestamp = self.generate_timestamp(slow_query_time, i)
            logs.append(
                f"{timestamp} [InventoryService] WARN: Slow query detected: SELECT * FROM inventory WHERE product_id IN (...) - execution time: {random.randint(3000, 7000)}ms"
            )

        logs.sort()
        with open(filename, "w") as f:
            for log in logs:
                f.write(log + "\n")

        print(f"Generated inventory service log: {filename}")


def main():
    """CLI interface for the log generator."""
    parser = argparse.ArgumentParser(description="Generate sample logs for testing")
    parser.add_argument(
        "-o",
        "--output-dir",
        default="sample_logs",
        help="Output directory for log files",
    )
    parser.add_argument(
        "-a", "--app-log", action="store_true", help="Generate application log"
    )
    parser.add_argument(
        "-e", "--error-log", action="store_true", help="Generate high error rate log"
    )
    parser.add_argument(
        "-m", "--microservices", action="store_true", help="Generate microservice logs"
    )
    parser.add_argument("--all", action="store_true", help="Generate all log types")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    generator = LogGenerator()

    print(f"Generating sample logs in directory: {output_dir}")

    if args.all or args.app_log:
        app_log_file = output_dir / "application.log"
        generator.generate_application_log(
            str(app_log_file), duration_hours=2, error_rate=0.08
        )

    if args.all or args.error_log:
        error_log_file = output_dir / "high_error_rate.log"
        generator.generate_high_error_rate_log(str(error_log_file), duration_minutes=30)

    if args.all or args.microservices:
        microservice_base = output_dir / "microservice"
        generator.generate_microservice_logs(str(microservice_base), num_services=3)

    # Generate a simple test log if no specific options
    if not any([args.app_log, args.error_log, args.microservices, args.all]):
        test_log_file = output_dir / "test.log"
        generator.generate_application_log(
            str(test_log_file), duration_hours=1, error_rate=0.1
        )

    print(f"\nSample logs generated successfully in {output_dir}/")
    print("You can now test the log analyzer with these files:")
    print(f"  python log_analyzer.py {output_dir}/application.log")
