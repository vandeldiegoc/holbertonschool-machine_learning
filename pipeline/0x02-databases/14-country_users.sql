
-- that creates a table users
CREATE TABLE IF NOT EXISTS  users (
    id INT NOT NULL AUTO_INCREMENT,
    email VARCHAR(254) NOT NULL UNIQUE,
    name VARCHAR(254),
    country ENUM('US', 'CO', 'TN') NOT NULL DEFAULT 'US',
    PRIMARY KEY(id)
);