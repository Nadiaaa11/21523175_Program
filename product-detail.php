<?php 
    require_once 'header.php';
    session_start();

    // Check if user is logged in and fetch user data
    if (isset($_SESSION['user_id'])) {
        $user_id = $_SESSION['user_id'];
        $user_check = $db->prepare("SELECT * FROM user WHERE user_id=:id");
        $user_check->execute(['id' => $user_id]);
        $userbring = $user_check->fetch(PDO::FETCH_ASSOC);
    } else {
        $userbring = null; // User is not logged in
    }

    // Fetch product details
    $sql = $db->prepare("SELECT * FROM product WHERE product_id=:product_id");
    $sql->execute(['product_id' => $_GET['product_id']]);
    $productbring = $sql->fetch(PDO::FETCH_ASSOC);
    $productcontrol = $sql->rowCount();
    
    if ($productcontrol == 0) {
        header("Location:index.php?status=empty");
        exit;
    }

    $product_id = $productbring['product_id'];

    $size_stock_sql = $db->prepare("SELECT size, stock FROM product_variant WHERE product_id=:product_id");
    $size_stock_sql->execute(['product_id' => $product_id]);
    $sizes = $size_stock_sql->fetchAll(PDO::FETCH_ASSOC);
?>
<head>
    <title><?php echo htmlspecialchars($productbring['product_name']); ?> - <?php echo htmlspecialchars($settingbring['setting_title']); ?></title>
</head>
<?php 
    if (isset($_GET['status']) && $_GET['status'] == "success") {
?>
    <script type="text/javascript">
        alert("Comment Added Successfully");
    </script>
<?php 
    }
?>
<div class="container">
    <div class="clearfix"></div>
    <div class="lines"></div>
</div>

<div class="container">
    <div class="row">
        <div class="col-md-9"><!--Main content-->
            <div class="title-bg">
                <div class="title"><?php echo htmlspecialchars($productbring['product_name']); ?></div>
            </div>
            <div class="row">
                <div class="col-md-6">
                    <?php
                        // Get the main product photo
                        $product_id = $productbring['product_id'];
                        $productphoto_control = $db->prepare("SELECT * FROM product_photo WHERE product_id=:product_id ORDER BY productphoto_order ASC LIMIT 1");
                        $productphoto_control->execute(['product_id' => $product_id]);
                        $productphoto_bring = $productphoto_control->fetch(PDO::FETCH_ASSOC);
                    ?>
                    <div class="dt-img">
                        <div class="detpricetag"><div class="inner">IDR <?php echo htmlspecialchars($productbring['product_price']); ?></div></div>
                        <a class="fancybox" href="<?php echo htmlspecialchars($productphoto_bring['productphoto_path']); ?>" data-fancybox-group="gallery" title="Cras neque mi, semper leon"><img src="<?php echo htmlspecialchars($productphoto_bring['productphoto_path']); ?>" alt="" class="img-responsive"></a>
                    </div>

                    <div class="thumb-gallery">
                        <h3>Color Choices</h3>
                        <?php
                            // Prepare to get distinct colors for this product
                            $color_control = $db->prepare("
                                SELECT DISTINCT color FROM product_variant 
                                WHERE product_id=:product_id 
                                ORDER BY color ASC
                            ");
                            $color_control->execute(['product_id' => $product_id]);

                            // Fetch and display each color with associated stock
                            while ($color_bring = $color_control->fetch(PDO::FETCH_ASSOC)) {
                                $color = htmlspecialchars($color_bring['color']); // Sanitize color name

                                // Fetch a photo associated with this color (if any)
                                $photo_control = $db->prepare("
                                    SELECT productphoto_path FROM product_photo 
                                    WHERE product_id=:product_id AND color=:color LIMIT 1
                                ");
                                $photo_control->execute([
                                    'product_id' => $product_id,
                                    'color' => $color
                                ]);
                                $photo_bring = $photo_control->fetch(PDO::FETCH_ASSOC);

                                // If no specific color image is found, use a default placeholder or fallback to the product thumbnail
                                if ($photo_bring) {
                                    $image_path = htmlspecialchars($photo_bring['productphoto_path']);
                                } else {
                                    // Fetch the product's thumbnail as a fallback if no color image is found
                                    $thumbnail_control = $db->prepare("
                                        SELECT productphoto_path FROM product_photo 
                                        WHERE product_id=:product_id AND productphoto_order = 1 LIMIT 1
                                    ");
                                    $thumbnail_control->execute(['product_id' => $product_id]);
                                    $thumbnail_bring = $thumbnail_control->fetch(PDO::FETCH_ASSOC);
                                    
                                    $image_path = $thumbnail_bring ? htmlspecialchars($thumbnail_bring['productphoto_path']) : 'default-image.jpg';
                                }
                        ?>
                        <div class="thumb-img">
                            <!-- Display the color as a clickable option -->
                            <a href="#" class="color-choice" data-color="<?php echo $color; ?>">
                                <img src="<?php echo $image_path; ?>" alt="<?php echo $color; ?>" class="img-responsive">
                                <p><?php echo $color; ?></p>
                            </a>
                            <!-- Optionally, display stock info if required -->
                        </div>
                        <?php } // End while loop ?>
                    </div>

                </div>
                <div class="col-md-6 det-desc">
                    <div class="productdata">
                        <div class="infospan">Code <span><?php echo htmlspecialchars($productbring['product_id']); ?></span></div>
                        <div class="infospan">Price <span>IDR <?php echo htmlspecialchars($productbring['product_price']); ?></span></div>

                        <form action="admin/system/work.php" method="POST" class="form-horizontal ava" role="form">
                            <div class="form-group">
                                <label for="qty" class="col-sm-2 control-label">Qty</label>
                                <div class="col-sm-4">
                                    <input type="text" class="form-control" value="1" name="product_unit">
                                </div>
                                <input type="hidden" name="user_id" value="<?php echo isset($userbring['user_id']) ? htmlspecialchars($userbring['user_id']) : ''; ?>">
                                <input type="hidden" name="product_id" value="<?php echo htmlspecialchars($productbring['product_id']); ?>">
                                <div class="col-sm-4">
                                    <?php if(isset($userbring)) { ?>
                                        <button type="submit" name="addtocart" class="btn btn-default btn-red btn-sm"><span class="addchart">Add To Cart</span></button>
                                    <?php } else { ?>
                                        <button type="submit" disabled class="btn btn-default btn-red btn-sm"><span class="addchart">Login</span></button>
                                    <?php } ?>
                                </div>
                                <div class="clearfix"></div>
                            </div>
                        </form>

                        <div class="sharing">
                            <div class="share-bt">
                                <div class="addthis_toolbox addthis_default_style ">
                                    <a class="addthis_counter addthis_pill_style"></a>
                                </div>
                                <script type="text/javascript" src="http://s7.addthis.com/js/250/addthis_widget.js#pubid=xa-4f0d0827271d1c3b"></script>
                                <div class="clearfix"></div>
                            </div>
                            <div class="avatock"><span>
                                <?php 
                                    if ($productbring['product_stock'] >= 1) {
                                        echo "In Stock : " . htmlspecialchars($productbring['product_stock']);
                                    } else {
                                        echo "Out of Stock";
                                    }
                                ?>
                            </span></div>
                        </div>
                        <h5>Sizes: </h5>
                        <select id="product_size" class="form-control">
                            <option value="">Select Size</option>
                            <?php foreach ($sizes as $size) { ?>
                                <option value="<?php echo htmlspecialchars($size['size']); ?>">
                                    <?php echo htmlspecialchars($size['size']); ?>
                                </option>
                            <?php } ?>
                        </select>
                        <div id="stock_info">
                            <!-- Stock info will be dynamically updated here -->
                        </div>
                    </div>
                </div>
            </div>

            <!-- Reviews section -->
            <?php
                $product_id = $productbring['product_id'];
                $commentsql = $db->prepare("SELECT * FROM comments WHERE product_id=:product_id AND comment_status=:comment_status");
                $commentsql->execute([
                    'product_id' => $product_id,
                    'comment_status' => 1
                ]);
            ?>

            <div class="tab-review">
                <ul id="myTab" class="nav nav-tabs shop-tab">
                    <li <?php if (!isset($_GET['status']) || $_GET['status'] != "success") {?> class="active" <?php } ?>><a href="#desc" data-toggle="tab">Description</a></li>
                    <li <?php if (isset($_GET['status']) && $_GET['status'] == "success") {?> class="active" <?php } ?>><a href="#rev" data-toggle="tab">Reviews (<?php echo $commentsql->rowCount(); ?>)</a></li>
                    <li class=""><a href="#video" data-toggle="tab">Video</a></li>
                </ul>
                <div id="myTabContent" class="tab-content shop-tab-ct">
                    <div class="tab-pane fade <?php if (!isset($_GET['status']) || $_GET['status'] != "success") {?> active in <?php } ?>" id="desc">
                        <?php echo htmlspecialchars($productbring['product_detail']); ?>
                    </div>
                    <div class="tab-pane fade <?php if (isset($_GET['status']) && $_GET['status'] == "success") {?> active in <?php } ?>" id="rev">
                        <?php
                            while ($commentbring = $commentsql->fetch(PDO::FETCH_ASSOC)) {
                                $commentuser_id = $commentbring['user_id'];
                                $commentuser_check = $db->prepare("SELECT * FROM user WHERE user_id=:id");
                                $commentuser_check->execute(['id' => $commentuser_id]);
                                $commentuserbring = $commentuser_check->fetch(PDO::FETCH_ASSOC);
                        ?>
                            <p class="dash">
                                <span><?php echo htmlspecialchars($commentuserbring['user_name']); ?></span> (<?php echo htmlspecialchars($commentbring['comment_time']); ?>)<br><br>
                                <?php echo htmlspecialchars($commentbring['comment_detail']); ?>
                            </p>
                        <?php } ?>

                        <?php if(isset($userbring)) { ?>
                            <h4>Write Review</h4>
                            <form action="admin/system/work.php" method="POST" role="form">
                                <div class="form-group">
                                    <textarea name="comment_detail" class="form-control" id="text"></textarea>
                                </div>
                                <input type="hidden" name="user_id" value="<?php echo isset($userbring['user_id']) ? htmlspecialchars($userbring['user_id']) : ''; ?>">
                                <input type="hidden" name="product_id" value="<?php echo htmlspecialchars($productbring['product_id']); ?>">
                                <button type="submit" class="btn btn-default btn-red btn-sm" name="addcomment">Submit</button>
                            </form>
                        <?php } else { ?>
                            <p>Please login to write a review.</p>
                        <?php } ?>
                    </div>
                    <div class="tab-pane fade" id="video">
                        <p>Product demo video here.</p>
                    </div>
                </div>
            </div>

            <div class="spacer"></div>
        </div><!--Main content-->
        <?php require_once 'sidebar.php'; ?>
    </div>
</div>

<?php require_once 'footer.php'; ?>

<script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
<script type="text/javascript">
    $(document).ready(function() {
        $('#product_size').change(function() {
            var selectedSize = $(this).val();
            var productId = '<?php echo $productbring['product_id']; ?>';

            if (selectedSize) {
                $.ajax({
                    url: 'get_stock.php',
                    type: 'GET',
                    data: { size: selectedSize, product_id: productId },
                    success: function(response) {
                        $('#stock_info').html('Stock: ' + response);
                    }
                });
            } else {
                $('#stock_info').html('');
            }
        });
    });
</script>
