<?php 
require_once 'header.php';
$sql=$db->prepare("SELECT * FROM about WHERE about_id=:id");
$sql->execute(['id' => 1]);
$aboutbring=$sql->fetch(PDO::FETCH_ASSOC);
?>
	<head>
		<title><?php echo $settingbring['setting_title']; ?></title>
	</head>
	<div class="container">

		<div class="clearfix"></div>
		<div class="lines"></div>

		<?php require_once 'slider.php'; ?>


	</div>
	<div class="f-widget featpro">
		<div class="container">
			<div class="title-widget-bg">
				<div class="title-widget">Featured Products</div>
				<div class="carousel-nav">
					<a class="prev"></a>
					<a class="next"></a>
				</div>
			</div>
			<div id="product-carousel" class="owl-carousel owl-theme">
				<?php 
					$productcheck=$db->prepare("SELECT * FROM product WHERE product_status=:product_status and product_homeshowcase=:product_homeshowcase");
					$productcheck->execute([
						'product_status' => 1,
						'product_homeshowcase' => 1,
						]);
					while ($productbring=$productcheck->fetch(PDO::FETCH_ASSOC)) {
						$product_id=$productbring['product_id'];
						$productphoto_control=$db->prepare("SELECT * FROM product_photo where product_id=:product_id order by productphoto_order ASC limit 1");
						$productphoto_control->execute([
							'product_id' => $product_id
						]);
						$productphoto_bring=$productphoto_control->fetch(PDO::FETCH_ASSOC);
				?>

				<div class="item">
					<div class="productwrap" data-description="<?php echo htmlspecialchars($productbring['product_detail']); ?>"> <!-- Store product description as a data attribute -->
						<div class="pr-img">
							<div class="hot"></div>
							<a href="product-<?=seo($productbring["product_name"]).'-'.$productbring["product_id"]?>">
								<img src="<?php echo $productphoto_bring['productphoto_path']; ?>" alt="" class="img-responsive">
							</a>
						</div>
						<span class="smalltitle">
							<a href="product-<?=seo($productbring["product_name"]).'-'.$productbring["product_id"]?>">
								<?php echo $productbring['product_name']; ?>
							</a>
						</span>
						<div class="pricetag blue">
							<div class="inner"><span>IDR <?php echo $productbring['product_price']; ?></span></div>
						</div>
						<!-- <span class="smalldesc">Item no. <?php echo $productbring['product_id']; ?></span> -->

						<!-- Hidden product description, initially not visible -->
						<!-- <div class="product-description" style="display: none;">
							<?php echo $productbring['product_detail']; ?>
						</div> -->
					</div>
				</div>

				<?php } ?>

			</div>
		</div>
	</div>
	
	
	
	<div class="container">
		<div class="row">
			<div class="col-md-9"><!--Main content-->
				<div class="title-bg">
					<div class="title"><?php echo $aboutbring['about_title']; ?></div>
				</div>
				<p class="ct">
					<?php echo substr($aboutbring['about_content'],0,1000); ?>
				</p>
				<a href="about" class="btn btn-default btn-red btn-sm">Read More</a>
				
				<div class="title-bg">
					<div class="title">Lastest Products</div>
				</div>
				<div class="row prdct"><!--Products-->
					<div class="col-md-4">
						<div class="productwrap" data-product-id="6"> <!-- Add data attribute for product ID -->
							<div class="pr-img">
								<a href="product.html"><img src="images\product\chinos.jpg" alt="" class="img-responsive"></a>
							</div>
							<span class="smalltitle"><a href="product.html">chinos</a></span>
							<div class="pricetag-below">
								<span class="oldprice">IDR 99000</span>
								<span class="newprice">IDR 45000</span>
							</div>
							<span class="smalldesc">Item no.: 6</span>
						</div>
					</div>
					<div class="col-md-4">
						<div class="productwrap" data-product-id="10"> <!-- Update product ID accordingly -->
							<div class="pr-img">
								<a href="product.htm"><img src="images\product\denim-jacket.jpg" alt="" class="img-responsive"></a>
								<div class="pricetag"><div class="inner">$199</div></div>
							</div>
							<span class="smalltitle"><a href="product.htm">Denim jacket</a></span>
							<span class="smalldesc">Item no.: 10</span>
						</div>
					</div>
					<!-- Add more products as needed -->
				</div><!--Products-->
				<!-- Product description area (hidden by default) -->
				<div id="product-description" style="display: none; position: absolute; background: #fff; padding: 10px; border: 1px solid #ccc;"></div>
				
				<div class="spacer"></div>
			</div><!--Main content-->
			<?php require_once 'sidebar.php'; ?>
		</div>
	</div>

<?php 
require_once 'footer.php';
?>

<script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
<script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.bundle.min.js"></script>

<script>
	$(document).ready(function() {
		// Show the popup when the page is loaded
		$('.popup-overlay, .popup-card').fadeIn();

		// Close the popup when the close button or overlay is clicked
		$('.close-btn, .popup-overlay').click(function() {
			$('.popup-overlay, .popup-card').fadeOut();
		});
	});

	$(document).ready(function(){
		// When hovering over the productwrap div, show the product description
		$('.productwrap').hover(function() {
			// Find the hidden description and display it
			$(this).find('.product-description').slideDown();
		}, function() {
			// On mouse leave, hide the description again
			$(this).find('.product-description').slideUp();
		});
	});

	$(document).ready(function(){
		// When hovering over the productwrap div, fetch and display the product description
		$('.productwrap').hover(function(event) {
			var productId = $(this).data('product-id'); // Get the product ID
			var productWrap = $(this);

			// Use AJAX to fetch the product description
			$.ajax({
				url: 'fetch_product_description.php', // A separate PHP file to handle fetching the product description
				method: 'POST',
				data: { product_id: productId },
				success: function(response) {
					// On success, show the product description
					$('#product-description').html(response).css({
						'top': event.pageY + 10,  // Position the description near the mouse
						'left': event.pageX + 10
					}).fadeIn();
				}
			});
		}, function() {
			// On mouse leave, hide the description
			$('#product-description').fadeOut();
		});

		// Move description area based on mouse movement
		$('.productwrap').mousemove(function(event) {
			$('#product-description').css({
				'top': event.pageY + 10,
				'left': event.pageX + 10
			});
		});
	});
</script>