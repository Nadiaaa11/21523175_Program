<?php
require_once 'admin/system/connect.php'; // Include your database connection file

if (isset($_GET['size']) && isset($_GET['product_id'])) {
    $size = $_GET['size'];
    $product_id = $_GET['product_id'];

    // Fetch stock for the selected size and product
    $stock_sql = $db->prepare("SELECT stock FROM product_variant WHERE product_id=:product_id AND size=:size");
    $stock_sql->execute(['product_id' => $product_id, 'size' => $size]);
    $stock_bring = $stock_sql->fetch(PDO::FETCH_ASSOC);

    if ($stock_bring) {
        echo $stock_bring['stock']; // Return stock for the size
    } else {
        echo 'Out of stock'; // If no stock is found
    }
}
?>
