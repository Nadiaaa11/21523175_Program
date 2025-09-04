<?php 
require_once 'header.php';

$onpage = 15; // Amount of content to be displayed on the page

// Default page number
$pages = isset($_GET['pages']) ? (int)$_GET['pages'] : 1;
if ($pages < 1) $pages = 1;

if (isset($_GET['sef'])) {
    // Fetch category details
    $categorycontrol = $db->prepare("SELECT * FROM categories WHERE category_seourl=:seourl");
    $categorycontrol->execute([
        'seourl' => $_GET['sef']
    ]);
    $categorybring = $categorycontrol->fetch(PDO::FETCH_ASSOC);

    if ($categorybring) {
        $category_id = $categorybring['category_id'];

        // Fetch total number of products for the selected category
        $totalProductQuery = $db->prepare("SELECT COUNT(*) as total FROM product WHERE category_id = :category_id");
        $totalProductQuery->execute([
            'category_id' => $category_id
        ]);
        $totalProduct = $totalProductQuery->fetch(PDO::FETCH_ASSOC)['total'];

        // Calculate total pages and limit
        $total_page = ceil($totalProduct / $onpage);
        if ($pages > $total_page) $pages = $total_page;
        $limit = ($pages - 1) * $onpage;

        // Fetch products with limit
        $productcontrol = $db->prepare("SELECT * FROM product WHERE category_id = :category_id ORDER BY product_order ASC LIMIT :limit, :onpage");
        $productcontrol->bindParam(':category_id', $category_id, PDO::PARAM_INT);
        $productcontrol->bindParam(':limit', $limit, PDO::PARAM_INT);
        $productcontrol->bindParam(':onpage', $onpage, PDO::PARAM_INT);
        $productcontrol->execute();
    } else {
        echo "<div class='col-md-12'>Category not found.</div>";
        exit;
    }
} else {
    // Fetch total number of products for all categories
    $totalProductQuery = $db->prepare("SELECT COUNT(*) as total FROM product");
    $totalProductQuery->execute();
    $totalProduct = $totalProductQuery->fetch(PDO::FETCH_ASSOC)['total'];

    // Calculate total pages and limit
    $total_page = ceil($totalProduct / $onpage);
    if ($pages > $total_page) $pages = $total_page;
    $limit = ($pages - 1) * $onpage;

    // Fetch all products with limit
    $productcontrol = $db->prepare("SELECT * FROM product ORDER BY product_order ASC LIMIT :limit, :onpage");
    $productcontrol->bindParam(':limit', $limit, PDO::PARAM_INT);
    $productcontrol->bindParam(':onpage', $onpage, PDO::PARAM_INT);
    $productcontrol->execute();
}

// Check if any products exist
$check = $productcontrol->rowCount();
if ($check == 0) {
    echo "<div class='col-md-12'>No products found in this category.</div>";
}
?>
<head>
    <title>Categories <?php echo isset($categorybring['category_name']) ? $categorybring['category_name'] : ''; ?> - <?php echo $settingbring['setting_title']; ?></title>
</head>
<div class="container">
    <div class="clearfix"></div>
    <div class="lines"></div>
</div>

<div class="container">
    <div class="row">
        <div class="col-md-9"><!--Main content-->
            <div class="title-bg">
                <div class="title">Products</div>
            </div>
            <div class="row prdct"><!--Products-->
                <?php
                while ($productbring = $productcontrol->fetch(PDO::FETCH_ASSOC)) {
                    $product_id = $productbring['product_id'];
                    $productphoto_control = $db->prepare("SELECT * FROM product_photo WHERE product_id=:product_id ORDER BY productphoto_order ASC LIMIT 1");
                    $productphoto_control->execute([
                        'product_id' => $product_id
                    ]);
                    $productphoto_bring = $productphoto_control->fetch(PDO::FETCH_ASSOC);
                ?>
                <div class="col-md-4">
                    <div class="productwrap">
                        <div class="pr-img">
                            <div class="hot"></div>
                            <a href="product-<?= seo($productbring["product_name"]) . '-' . $productbring["product_id"] ?>"><img src="<?php echo $productphoto_bring['productphoto_path']; ?>" alt="" class="img-responsive"></a>
                        </div>
                        <span class="smalltitle"><a href="#"><?php echo $productbring['product_name'] ?></a></span>
                        <div class="pricetag on-sale">
                            <div class="inner">
                                <span class="oldprice">IDR <?php echo $productbring['product_price'] * 2; ?></span>
                                <span class="newprice">IDR <?php echo $productbring['product_price']; ?></span>
                            </div>
                        </div>
                    </div>
                </div>
                <?php } ?>

                <!--Pagination-->
                <div align="right" class="col-md-12">
                    <ul class="pagination">
                        <!-- First Page Button -->
                        <li class="<?php echo ($pages == 1) ? 'disabled' : ''; ?>">
                            <a href="?sef=<?php echo $_GET['sef'] ?? ''; ?>&pages=1">First</a>
                        </li>

                        <!-- Current Page out of Total Pages -->
                        <li class="active">
                            <span>Page <?php echo $pages; ?> of <?php echo $total_page; ?></span>
                        </li>

                        <!-- Next Page Button -->
                        <li class="<?php echo ($pages == $total_page) ? 'disabled' : ''; ?>">
                            <a href="?sef=<?php echo $_GET['sef'] ?? ''; ?>&pages=<?php echo $pages + 1; ?>">Next</a>
                        </li>
                    </ul>
                </div>
                <!--Pagination-->

            </div><!--Products-->
        </div>
        <?php require_once 'sidebar.php'; ?>
    </div>
    <div class="spacer"></div>
</div>

<?php 
require_once 'footer.php';
?>
