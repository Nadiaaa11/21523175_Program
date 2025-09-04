<?php
require_once 'admin/system/connect.php'; // Include your database connection

if (isset($_POST['product_id'])) {
    $product_id = $_POST['product_id'];

    // Fetch product description from the database
    $sql = $db->prepare("SELECT product_description FROM product WHERE product_id = :product_id");
    $sql->execute(['product_id' => $product_id]);
    $product = $sql->fetch(PDO::FETCH_ASSOC);

    if ($product) {
        echo htmlspecialchars($product['product_description']); // Return the description
    } else {
        echo 'Description not available.';
    }
}
?>
